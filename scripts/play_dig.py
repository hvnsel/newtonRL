# scripts/play_dig.py
#
# Run a trained excavate policy (or a parked or scripted machine) with the
# Newton viewer, and write a per-step CSV trace of the mass balance and a
# heightmap dump for render_dig.py / render_attitude.py.
#
#   bash workstation/train.sh -p scripts/play_dig.py
#   bash workstation/train.sh -p scripts/play_dig.py --run 2026-10-24_11-20-38
#   bash workstation/train.sh -p scripts/play_dig.py --no_window --trace dig.csv
#
# Trace columns: removal_kg, payload_kg and cut_total_kg are the env's mass
# balance; below_kg is soil taken from under the plane.

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)
from luna_hifi_tasks.excavator.excavator import (
    ARM_LEN,
    CHASSIS_Z,
    DIG_DRUM_SIGN,
    MAST_TOP_Z,
    ROTOR_TIP_R,
    WHEEL_RADIUS,
)
from luna_hifi_tasks.excavator.excavator_cfg import MAX_DRUM_SPEED, MAX_WHEEL_SPEED, usd_status

TASK = "Luna-Excavator-Excavate"
LOG_ROOT = "logs/rsl_rl/excavator_excavate"


def _parse(argv):
    p = argparse.ArgumentParser(description="Play a trained excavate policy.")
    p.add_argument("--task", default=TASK, help=f"default {TASK}")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--seconds", type=float, default=25.0)
    p.add_argument("--run", default=None,
                   help=f"run directory under {LOG_ROOT}; default is the newest")
    p.add_argument("--checkpoint", default=None,
                   help="model file inside the run; default is the highest-numbered")
    p.add_argument("--no_window", action="store_true", help="skip the viewer")
    p.add_argument("--park", action="store_true",
                   help="no policy: wheels commanded to zero, arms held at the start angle, drums still")
    p.add_argument("--trace", default="dig_trace.csv", help="per-step CSV")
    p.add_argument("--frames", default=None,
                   help="directory for PNGs, if the env can render to an array")
    p.add_argument("--bed_depth", type=float, default=None, help="override bed depth [m]")
    p.add_argument("--mpm_voxel", type=float, default=None,
                   help="override the MPM solver grid cell [m]")
    p.add_argument("--young", type=float, default=None, help="override soil_young_modulus [Pa]")
    p.add_argument("--cohesion", type=float, default=None,
                   help="override soil_cohesion (yield_stress) [Pa]")
    p.add_argument("--gravity_scale", type=float, default=1.0, help="scale gravity")
    p.add_argument("--no_rails", dest="rails", action="store_false", default=None,
                   help="wheels on soil, rails parked")
    p.add_argument("--rails", dest="rails", action="store_true", default=None,
                   help="wheels on rigid rails at grade")
    p.add_argument("--wheel_mass_scale", type=float, default=None,
                   help="override wheel_proxy_mass_scale, a multiple of the 14.1 kg wheel")
    p.add_argument("--proxy_relaxation", type=float, default=None,
                   help="override proxy_relaxation, the rigid<->MPM force relaxation in [0, 1]")
    p.add_argument("--max_iterations", type=int, default=None,
                   help="override mpm_max_iterations (Gauss-Seidel sweeps per solve); "
                        "a multiple of 25")
    p.add_argument("--tolerance", type=float, default=None, help="override mpm_tolerance")
    p.add_argument("--mpm_verbose", action="store_true",
                   help="print the MPM solve's residuals and iteration count each step; "
                        "needs --no_cuda_graph")
    p.add_argument("--no_cuda_graph", action="store_true", help="disable CUDA graph capture")
    p.add_argument("--arm_effort", type=float, default=None,
                   help="arm actuator effort_limit_sim [N.m]")
    p.add_argument("--coupler_iterations", type=int, default=None,
                   help="override coupler_iterations, rigid<->MPM passes per physics step")
    p.add_argument("--proxy_relaxation_mode", default=None,
                   choices=("fixed", "aitken"),
                   help="override proxy_relaxation_mode")
    p.add_argument("--dig", action="store_true",
                   help="no policy: a straight pass at --dig_speed once the env has lowered the tool")
    p.add_argument("--straight", action="store_true",
                   help="zero the spawn yaw, xy, arm and gradient jitter")
    p.add_argument("--dig_speed", type=float, default=0.15,
                   help="forward command during --dig, a fraction of MAX_WHEEL_SPEED")
    # Tool overrides (the env lowers and spins the tool after the settle).
    p.add_argument("--cut", type=float, default=None,
                   help="override dig_cut: vane tips below the surface [m]")
    p.add_argument("--drum", type=float, default=None,
                   help="override dig_drum_speed as a fraction of MAX_DRUM_SPEED; "
                        "the sign of DIG_DRUM_SIGN loads")
    p.add_argument("--inlet_tilt", type=float, default=None,
                   help="override dig_inlet_tilt: inlet angle off straight down [rad]")
    p.add_argument("--dig_ramp", type=float, default=None,
                   help="override dig_ramp_seconds: time to lower the arms and spin the drums up")
    p.add_argument("--limp", action="store_true",
                   help="zero every joint's stiffness and damping, so nothing is driven")
    p.add_argument("--probe", action="store_true",
                   help="after the run, print the soil's contact impulses per body "
                        "and its stress state")
    p.add_argument("--dump", default="dig_bed.npz",
                   help="save the soil heightmap and body positions per frame here")
    p.add_argument("--dump_every", type=int, default=4, help="steps between saved frames")
    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=["newton_gl"])
    args, hydra_args = setup_preset_cli(p, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def _resolve_checkpoint(run: str | None, ckpt: str | None) -> Path:
    """The named or newest run directory, and the named or highest-numbered model_*.pt in it."""
    root = Path(LOG_ROOT)
    if run is None:
        run_dir = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)[-1]
    else:
        run_dir = root / run
    if ckpt is not None:
        return run_dir / ckpt
    return max(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))


def _load_policy(device: str, ckpt: Path):
    """The actor MLP from an rsl_rl checkpoint, as obs -> mean action.

    The actor's Linears are the 2-D weight tensors whose key names neither the
    critic, a normaliser nor the optimiser, ordered on the last integer in the
    key; widths come from the shapes, and ELU sits between them. A non-critic
    normaliser mean/var of the actor's input width is applied first.
    """
    import torch.nn as nn

    blob = torch.load(str(ckpt), map_location=device, weights_only=False)

    def walk(d, prefix=""):
        """Every tensor in a dict of dicts, keyed by its dotted path."""
        out = {}
        for k, v in d.items():
            path = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(walk(v, f"{path}."))
            elif hasattr(v, "dim"):
                out[path] = v
        return out

    flat = walk(blob)

    mats: dict[int, tuple[str, tuple[int, int]]] = {}
    for key, value in flat.items():
        low = key.lower()
        if value.dim() != 2 or not low.endswith("weight"):
            continue
        if "critic" in low or "norm" in low or "optimizer" in low:
            continue
        nums = re.findall(r"\d+", key)
        if not nums:
            continue
        mats[int(nums[-1])] = (key, tuple(value.shape))

    order = sorted(mats)
    print("[play] actor layers found:")
    for i in order:
        key, (out_f, in_f) = mats[i]
        print(f"         {key:40s} {in_f:4d} -> {out_f:4d}")

    layers: list[nn.Module] = []
    for n, idx in enumerate(order):
        out_f, in_f = mats[idx][1]
        layers.append(nn.Linear(in_f, out_f))
        if n != len(order) - 1:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)

    own = [i for i, m in enumerate(actor) if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for mine, idx in zip(own, order):
            wkey = mats[idx][0]
            actor[mine].weight.copy_(flat[wkey])
            actor[mine].bias.copy_(flat[wkey[: -len("weight")] + "bias"])
    actor.to(device).eval()

    # Normaliser stats, spelt mean/var or _mean/_var, shaped (1, D).
    mean = var = None
    for key, value in flat.items():
        low = key.lower()
        if "norm" not in low or "critic" in low:
            continue
        leaf = key.rsplit(".", 1)[-1].lstrip("_").lower()
        if leaf == "mean":
            mean = value.to(device).flatten()
        elif leaf == "var":
            var = value.to(device).flatten()

    in_dim = mats[order[0]][1][1]
    if mean is not None and mean.numel() != in_dim:
        print(f"[play] ignoring a normaliser of {mean.numel()} dims against an "
              f"actor input of {in_dim}")
        mean = var = None
    norm = f"mean/var over {mean.numel()} dims" if mean is not None else "none"
    print(f"[play] actor {in_dim} -> "
          f"{' -> '.join(str(mats[i][1][0]) for i in order)}, elu, obs norm: {norm}")
    print(f"[play] checkpoint iteration {blob.get('iter', 'unknown')}")

    def policy(obs: torch.Tensor) -> torch.Tensor:
        x = obs
        if mean is not None and var is not None:
            x = (x - mean) / torch.sqrt(var + 1e-8)
        return actor(x)

    return policy


def _attitude(u) -> None:
    """Print arm and shroud joint angles (achieved and, where the data exposes
    it, commanded) and each wheel's and drum's lowest point against the bed
    surface."""
    d = u.robot.data
    names = u.robot.joint_names
    q = d.joint_pos.torch[0].detach().cpu().numpy()
    tgt = getattr(d, "joint_pos_target", None)
    tgt = tgt.torch[0].detach().cpu().numpy() if tgt is not None else None
    print("[attitude] arm and shroud joints, commanded vs achieved:")
    for i, n in enumerate(names):
        if not ("arm" in n or "shroud" in n):
            continue
        c = f"{tgt[i]:+8.4f}" if tgt is not None else "       ?"
        print(f"[attitude]   {n:14s} q={q[i]:+8.4f} rad ({np.degrees(q[i]):+7.2f} deg)"
              f"  target={c}")

    bl = u.robot.body_names
    bz = d.body_pos_w.torch[0, :, 2].detach().cpu().numpy()
    want = [i for i, n in enumerate(bl) if "wheel" in n or "drum" in n]
    surf = float(u.cfg.bed_top)
    # A body's z is its axis; its lowest point is one wheel or rotor-tip radius below.
    print(f"[attitude] body heights (world z), bed surface z={surf:+.4f}:")
    for i in want:
        r = WHEEL_RADIUS if "wheel" in bl[i] else ROTOR_TIP_R
        bot = bz[i] - r
        print(f"[attitude]   {bl[i]:22s} axis z={bz[i]:+8.4f}  "
              f"bottom={bot:+8.4f}  {1000 * (bot - surf):+7.0f} mm "
              f"{'BELOW' if bot < surf else 'above'} the bed")
    wz = [bz[i] for i in want if "wheel" in bl[i]]
    print(f"[attitude]   wheel z spread {1000 * (max(wz) - min(wz)):.0f} mm")


def _probe(u) -> None:
    """Print, for each MPM coupler entry, the collider impulses per body, which
    body each active grid node is attributed to, and the soil's stress state.

    Body indices are local to the MPM entry's model view. A grid node is
    attributed to the single collider with the smallest sdf there, so n counts
    nodes a body won.
    """
    import newton

    coupled = u.sim.physics_manager._solver
    print(f"[probe] coupled solver: {type(coupled).__name__}")
    names = tuple(coupled.entry_names())
    print(f"[probe] entries: {names}")
    proxy = int(newton.BodyFlags.PROXY)
    kinflag = int(newton.BodyFlags.KINEMATIC)

    for name in names:
        solver = coupled.solver(name)
        kind = type(solver).__name__
        if not hasattr(solver, "collect_collider_impulses"):
            print(f"[probe] {name}: {kind}")
            continue
        print(f"[probe] {name}: {kind}  <- MPM")

        # --- impulses, net per body ---
        state = coupled.entry_state(name, "current")
        imp, pos, ids = solver.collect_collider_impulses(state)
        imp_np = imp.numpy()
        ids_np = ids.numpy()
        mag = np.linalg.norm(imp_np, axis=-1)
        print(f"[probe]   collider impulse entries {ids_np.shape[0]}, "
              f"valid ids {int((ids_np >= 0).sum())}")
        print(f"[probe]   sum of |impulse| {mag.sum():.4f} N.s (magnitudes, not net)")

        mdl = solver.model
        cbi = solver.collider_body_index.numpy()
        flags = mdl.body_flags.numpy()
        nb = int(flags.shape[0])
        net = np.zeros((nb, 3))
        count = np.zeros(nb, dtype=int)
        ok = (ids_np >= 0) & (ids_np < cbi.shape[0])
        bodies = cbi[ids_np[ok]]
        good = bodies >= 0
        np.add.at(net, bodies[good], imp_np[ok][good])
        np.add.at(count, bodies[good], 1)
        print("[probe]   net impulse per body (N.s), +z holds it up:")
        for b in range(nb):
            tag = "PROXY" if (flags[b] & proxy) else "kinematic"
            print(f"[probe]     body {b:2d} {tag:9s} n={count[b]:6d}  "
                  f"net=({net[b][0]:+8.3f},{net[b][1]:+8.3f},{net[b][2]:+9.3f})")
        zp = net[(flags & proxy) != 0][:, 2].sum()
        print(f"[probe]   proxy bodies net +z: {zp:+.3f} N.s")
        mass = float(u.robot.data.default_mass.torch[0].sum().item())
        weight = mass * float(np.linalg.norm(np.asarray(u.cfg.sim.gravity, dtype=float)))
        for dt_ in (0.01, 0.04):
            print(f"[probe]     as force at dt={dt_}: {zp / dt_:+9.1f} N "
                  f"(weight is {weight:.0f} N, {mass:.0f} kg)")

        # --- node attribution ---
        print(f"[probe]   collider_body_index {cbi.shape[0]} entries, "
              f"{int((cbi >= 0).sum())} map to a body")
        labels = [str(mdl.body_label[b]) for b in range(nb)]
        margin = mdl.shape_margin.numpy()
        sflags = mdl.shape_flags.numpy()
        scale = mdl.shape_scale.numpy()
        sxf = mdl.shape_transform.numpy()
        collide = int(newton.ShapeFlags.COLLIDE_PARTICLES)

        vox = float(solver._mpm_model.voxel_size)
        deg = int(solver._scratchpad.collider_position_field.degree)
        # Contact activation band: 0.5 / degree voxels, none for a degree-0 basis.
        act = 0.0 if deg == 0 else (0.5 / deg) * vox
        print(f"[probe]   voxel {vox:.3f} m, collider basis degree {deg}, "
              f"activation {act:.4f} m")
        bpos = state.body_q.numpy()[:, :3]

        # Per-body bounding radius over its particle-colliding shapes, body-local:
        # |shape offset| + largest half-extent.
        radius = np.zeros(nb)
        ngeom = np.zeros(nb, dtype=int)
        shape_ids = {}
        for b in range(nb):
            sl = [int(s) for s in mdl.body_shapes.get(b, ()) if sflags[int(s)] & collide]
            shape_ids[b] = sl
            ngeom[b] = len(sl)
            for s in sl:
                off = float(np.linalg.norm(sxf[s][:3]))
                radius[b] = max(radius[b], off + float(np.max(np.abs(scale[s]))))

        print("[probe]   per-body attribution; n is nodes won, not force:")
        print("[probe]      b tag       geoms       margin   reach "
              "     n    sum|f|     |net|   maxnode        z  name")
        reach = np.zeros(nb)
        for b in range(nb):
            mg = margin[shape_ids[b]] if shape_ids[b] else np.zeros(1)
            reach[b] = float(mg.max()) + act
            if count[b]:
                fb = imp_np[ok][good][bodies[good] == b]
                fm = np.linalg.norm(fb, axis=-1)
                smag, nmag = float(fm.sum()), float(np.linalg.norm(fb.sum(axis=0)))
                mx = float(fm.max())
            else:
                smag = nmag = mx = 0.0
            tag = ("PROXY" if flags[b] & proxy else
                   "KINEMATIC" if flags[b] & kinflag else "plain")
            print(f"[probe]     {b:2d} {tag:9s} {ngeom[b]:5d} "
                  f"{mg.min():.3f}-{mg.max():.3f}  {reach[b]:.4f} "
                  f"{count[b]:6d} {smag:9.3f} {nmag:9.3f} {mx:9.3f} "
                  f"{bpos[b][2]:+8.4f}  {labels[b]}")

        # Nodes inside a body's bounding sphere plus reach that another body won.
        apos = pos.numpy()[ok][good]
        awin = bodies[good]
        print("[probe]   contested nodes, inside one body's reach but won by another:")
        for b in range(nb):
            if ngeom[b] == 0:
                continue
            near = np.linalg.norm(apos - bpos[b], axis=1) <= radius[b] + reach[b]
            lost = near & (awin != b)
            if not lost.any():
                continue
            w, c = np.unique(awin[lost], return_counts=True)
            top = ", ".join(
                f"{labels[int(wi)].rsplit('/', 1)[-1]}({int(ci)})"
                for wi, ci in sorted(zip(w, c), key=lambda t: -t[1])[:4]
            )
            print(f"[probe]     {labels[b].rsplit('/', 1)[-1]:24s} "
                  f"kept {count[b]:6d}  lost {int(lost.sum()):6d}  to {top}")

        # --- soil stress ---
        F = state.mpm.particle_elastic_strain.numpy()   # elastic deformation gradient
        q = state.particle_q.numpy()
        E = solver.model.mpm.young_modulus.numpy()
        nu = solver.model.mpm.poisson_ratio.numpy()
        mu = solver.model.mpm.friction.numpy()
        ys = solver.model.mpm.yield_stress.numpy()

        diag = np.einsum("...ii->...i", F)
        dev_from_I = F - np.eye(3)
        print(f"[probe]   particle_elastic_strain raw: "
              f"diag mean {diag.mean():.6f} "
              f"min {diag.min():.4f} max {diag.max():.4f}")
        print(f"[probe]     |F - I| mean {np.abs(dev_from_I).mean():.6f} "
              f"max {np.abs(dev_from_I).max():.4f}  (E={E.mean():.1g} Pa)")

        # particle_stress is the solver's yield-projected stress [Pa],
        # compression-positive. Yield surface: sqrt(J2) <= yield_stress + friction * p.
        sg = state.mpm.particle_stress.numpy()
        pp = np.trace(sg, axis1=1, axis2=2) / 3.0
        dv = sg - pp[:, None, None] * np.eye(3)
        tt = np.sqrt(0.5 * np.einsum("nij,nij->n", dv, dv))
        lim = ys + mu * np.clip(pp, 0.0, None)
        rat = tt / np.maximum(lim, 1e-9)
        npart = int(pp.shape[0])
        floor = int((pp <= 0.0).sum())
        print(f"[probe]   particle_stress (compression-positive): "
              f"p mean {pp.mean():.1f}Pa "
              f"p1 {np.percentile(pp, 1):.1f}Pa "
              f"p50 {np.percentile(pp, 50):.1f}Pa "
              f"p99 {np.percentile(pp, 99):.1f}Pa")
        print(f"[probe]     sqrt(J2) mean {tt.mean():.1f}Pa, "
              f"yield = {ys.mean():.0f} + {mu.mean():.3f}*p, "
              f"tau/yield mean {rat.mean():.3f} "
              f"p99 {np.percentile(rat, 99):.3f}  (1.0 is the yield surface)")
        # p is clamped at 0. sigma == 0 identically marks a particle with no strain node.
        dead = int(((np.abs(pp) <= 0.0) & (tt <= 0.0)).sum())
        print(f"[probe]     p <= 0: {floor} of {npart} particles "
              f"= {100.0 * floor / max(npart, 1):.2f}%")
        print(f"[probe]     sigma == 0 identically: {dead} of {npart} "
              f"= {100.0 * dead / max(npart, 1):.2f}%")

        # Mean pressure within 0.35 m (xy) of each wheel whose axis is within
        # 0.20 m of the surface, against the rest, per depth band.
        bp = u.robot.data.body_pos_w.torch[0].detach().cpu().numpy()
        bn = u.robot.body_names
        org = u.scene.env_origins[0].detach().cpu().numpy()
        top = float(np.quantile(q[:, 2], 0.995))
        dep = top - q[:, 2]
        wi = [k for k, n in enumerate(bn) if "wheel" in n]
        low = [k for k in wi if (bp[k][2] - org[2]) < top + 0.20]
        if low:
            near = np.zeros(q.shape[0], dtype=bool)
            for k in low:
                c = bp[k][:2] - org[:2]
                near |= np.hypot(q[:, 0] - c[0], q[:, 1] - c[1]) < 0.35
            print("[probe]   pressure under the loaded wheels vs away, same depth:")
            print(f"[probe]     loaded: {[bn[k].replace('_body', '') for k in low]}")
            print(f"[probe]     {'band':>11} {'n near':>7} {'p near':>11} "
                  f"{'n far':>7} {'p far':>11} {'ratio':>6}")
            for lo, hi in ((0.0, 0.10), (0.10, 0.20), (0.20, 0.40)):
                m = (dep >= lo) & (dep < hi)
                a, b = m & near, m & ~near
                if a.sum() < 20 or b.sum() < 20:
                    continue
                pa, pb = float(pp[a].mean()), float(pp[b].mean())
                print(f"[probe]     {lo:.2f}-{hi:.2f}m {int(a.sum()):7d} "
                      f"{pa:10.1f}Pa {int(b.sum()):7d} {pb:10.1f}Pa "
                      f"{pa / max(pb, 1e-9):6.2f}")

        # Stress from F by linear elasticity on the log strain U log(S) U^T,
        # which discards F's rotation. Compression-positive pressure.
        Fu, Fs, _Fv = np.linalg.svd(F)
        eps = np.einsum("nik,nk,njk->nij", Fu, np.log(np.maximum(Fs, 1e-12)), Fu)
        tr = np.trace(eps, axis1=1, axis2=2)
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        two_g = E / (1 + nu)
        sig = two_g[:, None, None] * eps + (lam * tr)[:, None, None] * np.eye(3)
        trs = np.trace(sig, axis1=1, axis2=2)
        pres = -trs / 3.0
        dev = sig - (trs / 3.0)[:, None, None] * np.eye(3)
        tau = np.sqrt(0.5 * np.einsum("nij,nij->n", dev, dev))
        limit = ys + mu * np.clip(pres, 0.0, None)

        z = q[:, 2]
        top = float(np.percentile(z, 99.5))
        print(f"[probe]   soil stress over {z.shape[0]} particles, surface at z={top:.3f}")
        print(f"[probe]     {'depth':>8s} {'n':>7s} {'pressure':>12s} "
              f"{'geostatic':>11s} {'tau':>10s} {'tau/yield':>10s}")
        rho = u.cfg.soil_density
        g = abs(float(u.cfg.sim.gravity[2]))
        for lo, hi in ((0.00, 0.05), (0.05, 0.10), (0.10, 0.15),
                       (0.15, 0.20), (0.20, 0.30)):
            m = (z > top - hi) & (z <= top - lo)
            if m.sum() == 0:
                continue
            d = 0.5 * (lo + hi)
            ratio = tau[m] / np.maximum(limit[m], 1e-9)
            print(f"[probe]     {d*100:6.1f}cm {int(m.sum()):7d} "
                  f"{pres[m].mean():10.1f}Pa {rho*g*d:9.1f}Pa "
                  f"{tau[m].mean():8.1f}Pa {ratio.mean():10.3f}")

        print(f"[probe]   bodies {nb}, "
              f"PROXY {int((flags & proxy != 0).sum())}, "
              f"KINEMATIC {int((flags & kinflag != 0).sum())}")


def main(argv=None) -> int:
    args = _parse(argv)
    torch.manual_seed(0)

    ckpt = None
    if args.park:
        print("[play] parked: zero wheel command, arms at the start angle, no policy")
    elif args.dig:
        print("[play] scripted pass, no policy")
    else:
        ckpt = _resolve_checkpoint(args.run, args.checkpoint)
        print(f"[play] checkpoint {ckpt}")

    problem = usd_status()
    if problem is not None:
        print(f"\n[asset] {problem}\n")
        return 2

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs
    # The sparse grid is sized from max_num_envs; __post_init__ re-derives it below.
    env_cfg.max_num_envs = args.num_envs

    if args.bed_depth is not None:
        env_cfg.bed_depth = args.bed_depth
        print(f"[play] bed_depth set to {args.bed_depth} m")

    if args.mpm_voxel is not None:
        env_cfg.mpm_voxel_size = args.mpm_voxel
        print(f"[play] mpm_voxel_size set to {args.mpm_voxel} m")

    if args.young is not None:
        env_cfg.soil_young_modulus = args.young
        print(f"[play] soil_young_modulus set to {args.young:g} Pa")

    if args.cohesion is not None:
        env_cfg.soil_cohesion = args.cohesion
        print(f"[play] soil_cohesion set to {args.cohesion} Pa")

    for value, field, scale in ((args.cut, "dig_cut", 1.0), (args.drum, "dig_drum_speed", MAX_DRUM_SPEED),
                                (args.inlet_tilt, "dig_inlet_tilt", 1.0), (args.dig_ramp, "dig_ramp_seconds", 1.0)):
        if value is not None:
            setattr(env_cfg, field, value * scale)
            print(f"[play] {field} set to {value * scale:g}")
    if args.park:
        env_cfg.dig_tool = False

    if args.park or args.probe or args.dig:
        # Episode long enough that no reset lands inside the run.
        want = args.seconds + 2.0
        if env_cfg.episode_length_s < want:
            print(f"[play] episode_length_s {env_cfg.episode_length_s} -> {want} "
                  f"so no reset lands inside the run")
            env_cfg.episode_length_s = want

    if args.rails is not None:
        env_cfg.wheel_rails = args.rails
        print(
            f"[play] wheel_rails set to {args.rails}: wheels on "
            f"{'rigid rails' if args.rails else 'soil'}"
        )

    if args.straight:
        env_cfg.spawn_yaw_jitter = 0.0
        env_cfg.spawn_x_jitter = 0.0
        env_cfg.spawn_y_jitter = 0.0
        env_cfg.arm_start_jitter = 0.0
        env_cfg.cut_gradient_max = 0.0
        print("[play] STRAIGHT: yaw, xy, arm and gradient jitter all zeroed; "
              "machine spawns on the centreline pointed down +x")

    if args.max_iterations is not None:
        env_cfg.mpm_max_iterations = args.max_iterations
        print(f"[play] mpm_max_iterations set to {args.max_iterations:g} sweeps")
    if args.tolerance is not None:
        env_cfg.mpm_tolerance = args.tolerance
        print(f"[play] mpm_tolerance set to {args.tolerance:g}")

    if args.no_cuda_graph:
        env_cfg.use_cuda_graph = False
        print("[play] CUDA graph capture off")

    if args.arm_effort is not None:
        env_cfg.scene.excavator.actuators["arms"].effort_limit_sim = args.arm_effort
        print(f"[play] arm effort_limit_sim set to {args.arm_effort:g} N.m")

    if args.coupler_iterations is not None:
        env_cfg.coupler_iterations = args.coupler_iterations
        print(f"[play] coupler_iterations set to {args.coupler_iterations}")

    if args.proxy_relaxation is not None:
        env_cfg.proxy_relaxation = args.proxy_relaxation
        print(f"[play] proxy_relaxation set to {args.proxy_relaxation:g}")

    if args.proxy_relaxation_mode is not None:
        env_cfg.proxy_relaxation_mode = args.proxy_relaxation_mode
        print(f"[play] proxy_relaxation_mode set to {args.proxy_relaxation_mode}")

    if args.wheel_mass_scale is not None:
        env_cfg.wheel_proxy_mass_scale = args.wheel_mass_scale
        print(
            f"[play] wheel_proxy_mass_scale set to {args.wheel_mass_scale:g}: "
            f"each wheel reaches the MPM solve as "
            f"{args.wheel_mass_scale * 14.1:.0f} kg"
        )

    # Re-derive the grid, rails, proxy mappings and material from the overrides.
    env_cfg.__post_init__()

    if args.gravity_scale != 1.0:
        g = tuple(float(c) * args.gravity_scale for c in env_cfg.sim.gravity)
        env_cfg.sim.gravity = g
        print(f"[play] gravity scaled by {args.gravity_scale}: {g}")

    if not args.no_window:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        env_cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                show_particles=True,
                particle_color=(0.62, 0.55, 0.45),
                eye=(5.0, 5.0, 3.0),
                lookat=(0.5, 0.0, 0.3),
            )
        ]

    if "--device" in sys.argv:
        print(f"[play] sim device set to {args.device} (from --device)")
        env_cfg.sim.device = args.device
    args.device = env_cfg.sim.device
    env_cfg.validate()

    frame_dir = Path(args.frames) if args.frames else None
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if frame_dir else None)
        u = env.unwrapped
        if args.park or args.dig:
            # [forward, yaw]: forward is dig_forward * (1 + a0), so -1 holds the wheels.
            fixed = torch.zeros(u.num_envs, 2, device=u.device)
            fixed[:, 0] = -1.0 if args.park else args.dig_speed / u.cfg.dig_forward - 1.0
            if args.dig:
                angle = u._dig_arm
                drum = u.cfg.dig_drum_speed
                if drum * DIG_DRUM_SIGN < 0:
                    print(f"[play] *** drums at {drum:+.2f} rad/s is the DUMP "
                          f"direction. Loading is {DIG_DRUM_SIGN:+.0f} ***")
                # Rotor tip height with the chassis at bed_top + CHASSIS_Z + 0.02 (SPAWN_Z).
                chassis0 = float(u.cfg.bed_top) + CHASSIS_Z + 0.02
                tip = chassis0 + MAST_TOP_Z - ARM_LEN * math.sin(angle) - ROTOR_TIP_R
                print(f"[play] DIG: arm {angle:+.4f} rad ({math.degrees(angle):+.1f} deg), "
                      f"shroud {u._dig_shroud:+.4f} rad, drums {drum:+.2f} rad/s; rotor tip "
                      f"{1000 * (tip - float(u.cfg.bed_top)):+.0f} mm from the bed surface; "
                      f"forward {args.dig_speed * MAX_WHEEL_SPEED * WHEEL_RADIUS:.2f} m/s "
                      f"after {u.cfg.settle_seconds:.1f} s settle + {u.cfg.dig_ramp_seconds:.1f} s ramp")
            k = [0]
            ramp_end = u._settle_steps + u._ramp_steps

            def policy(_obs, _k=k):
                _k[0] += 1
                if args.dig and _k[0] == ramp_end + 1:
                    print("[play] --- end of dig ramp, before moving off ---")
                    _attitude(u)
                    print(f"[play] undisturbed bed surface is z={u.cfg.bed_top:.4f}")
                return fixed
        else:
            policy = _load_policy(str(u.device), ckpt)

        if args.limp:
            nj = u.robot.data.joint_pos.torch.shape[1]
            z = torch.zeros((u.num_envs, nj), device=u.device)
            u.robot.write_joint_stiffness_to_sim(z)
            u.robot.write_joint_damping_to_sim(z)
            print(f"[play] LIMP: stiffness and damping zeroed on {nj} joints")

        if args.mpm_verbose:
            # The MPM solver reads self.verbose at each solve.
            coupled = u.sim.physics_manager._solver
            hit = 0
            for nm in coupled.entry_names():
                sv = coupled.solver(nm)
                if hasattr(sv, "collect_collider_impulses"):
                    sv.verbose = True
                    hit += 1
                    print(f"[play] MPM verbose on for entry {nm!r}")
            if hit == 0:
                print("[play] --mpm_verbose found no MPM solver; nothing set")

        dt = u.step_dt
        steps = int(args.seconds / dt)
        reset_steps: list[int] = []
        bodies: list = []
        t_start = time.perf_counter()
        t_mark = t_start
        soil = u.cfg.soil_density
        print(f"[play] {steps} steps of {dt:.4f}s, {u.cfg.bed_particles_per_env} particles/env")

        def scalar(name: str) -> float:
            """Env 0's value of an env attribute."""
            return float(getattr(u, name)[0])

        # Per saved frame: (ny, nx) env-local bed heights, and
        # [x, y, z, yaw, t, removal, payload] of the chassis.
        beds: list = []
        poses: list = []

        rows: list[dict] = []
        obs, _ = env.reset()
        actor_obs = obs["policy"]
        frames_written = 0
        frames_off = frame_dir is None

        for i in range(steps):
            # Read before the step: the reset on the step a pass ends zeroes it.
            fill_prev = u._fill_kg[0].tolist()
            with torch.inference_mode():
                action = policy(actor_obs).clamp(-1.0, 1.0)
                obs, _, term, trunc, _ = env.step(action)
                actor_obs = obs["policy"]
                dones = term | trunc

            root = u.robot.data.root_pos_w.torch
            below = scalar("_below_prev") * soil
            removal = scalar("_removed_kg")
            rows.append({
                "t": round(i * dt, 4),
                "removal_kg": round(removal, 3),
                "payload_kg": round(scalar("_payload_kg"), 3),
                "cut_total_kg": round(scalar("_cut_total_kg"), 3),
                "below_kg": round(below, 3),
                "advance_mps": round(float(u.robot.data.root_lin_vel_b.torch[0, 0]), 4),
                "chassis_z": round(float(root[0, 2] - u.scene.env_origins[0, 2]), 4),
                "particle_max_m": round(scalar("_particle_max_m"), 2),
                "shape_done": int(scalar("_shape_done")),
                "escaped": int(scalar("_escaped")),
            })

            if args.dump and i % args.dump_every == 0:
                beds.append(u._bed_grid[0].detach().float().cpu().numpy().copy())
                origin = u.scene.env_origins[0]
                bp = u.robot.data.body_pos_w.torch[0].detach().cpu().numpy()
                bodies.append((bp - origin.detach().cpu().numpy()).copy())
                poses.append([
                    float(root[0, 0] - origin[0]),
                    float(root[0, 1] - origin[1]),
                    float(root[0, 2] - origin[2]),
                    float(u._base_yaw()[0]),
                    float(i * dt),
                    removal,
                    scalar("_payload_kg"),
                ])

            if frame_dir is not None:
                # The Newton backend returns no RGB buffer; frames stop at the first miss.
                try:
                    img = env.render()
                except Exception as exc:                      # noqa: BLE001
                    print(f"[play] no RGB buffer from this backend ({exc.__class__.__name__}: "
                          f"{exc}); screen-record the viewer window instead")
                    frame_dir, frames_off = None, True
                else:
                    if img is None:
                        print("[play] render() returned None; screen-record the viewer instead")
                        frame_dir, frames_off = None, True
                    else:
                        from PIL import Image

                        Image.fromarray(img).save(frame_dir / f"{i:05d}.png")
                        frames_written += 1

            if (i + 1) % 100 == 0:
                now = time.perf_counter()
                sps = 100.0 / (now - t_mark)
                t_mark = now
                print(f"[play] step {i + 1}/{steps}  {sps:6.2f} env-steps/s  "
                      f"eta {(steps - i - 1) / sps:6.0f}s", flush=True)

            if bool(dones.any()):
                reset_steps.append(i)
                print(f"[play] t={i*dt:5.2f}s episode ended "
                      f"(pass_done={int(scalar('_pass_done'))}, "
                      f"shape_done={int(scalar('_shape_done'))}, "
                      f"escaped={int(scalar('_escaped'))}), "
                      f"in the drums: front {fill_prev[0]:.1f} kg, rear {fill_prev[1]:.1f} kg")

        if args.probe:
            _attitude(u)
            if reset_steps:
                since = steps - 1 - reset_steps[-1]
                print(
                    f"\n[probe] WARNING: the episode reset {since} step(s) before "
                    f"this probe (t={reset_steps[-1]*dt:.2f}s); the readings below "
                    f"describe a freshly reset scene."
                )
            _probe(u)

        if args.dump and beds:
            np.savez_compressed(
                args.dump,
                bed=np.stack(beds).astype("float32"),
                pose=np.asarray(poses, dtype="float32"),
                work_mask=u._work_mask[0].detach().cpu().numpy(),
                body_pos=np.stack(bodies).astype("float32"),
                # Unicode, readable without allow_pickle.
                body_names=np.asarray(u.robot.body_names, dtype="U64"),
                wheel_radius=WHEEL_RADIUS,
                voxel=float(u.cfg.voxel_size),
                lower=np.asarray(u.cfg.bed_grid_lower, dtype="float32"),
                bed_x=np.asarray(u.cfg.bed_x, dtype="float32"),
                bed_y=np.asarray(u.cfg.bed_y, dtype="float32"),
                cut_ref_kg=float(u.cfg.cut_volume_ref * soil),
                drum_capacity_kg=float(2.0 * u.cfg.target_load_kg),
            )
            print(f"[play] wrote {len(beds)} heightmap frames to {args.dump}")
            print(f"[play] render with: python3 scripts/render_dig.py {args.dump}")

        with open(args.trace, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[play] wrote {len(rows)} steps to {args.trace}")
        if frames_written:
            print(f"[play] wrote {frames_written} PNGs to {args.frames}")
        elif not frames_off:
            print("[play] no frames written")

        wall = time.perf_counter() - t_start
        sps = steps / wall
        print(f"[play] {steps} steps in {wall:.1f}s = {sps:.2f} env-steps/s "
              f"at {u.num_envs} env(s), {sps * dt:.2f}x realtime")

        last = rows[-1]
        peak = max(r["payload_kg"] for r in rows)

        z = [r["chassis_z"] for r in rows]
        tail = z[-min(50, len(z)):]
        rate = (tail[0] - tail[-1]) / (max(len(tail) - 1, 1) * dt)
        print(f"[play] chassis_z {z[0]:.4f} -> {z[-1]:.4f} m, min {min(z):.4f}: "
              f"sank {1000 * (z[0] - min(z)):.0f} mm, "
              f"last {len(tail)} steps {1000 * (tail[0] - tail[-1]):+.1f} mm "
              f"({1000 * rate:+.2f} mm/s)")

        print(f"[play] end: removal {last['removal_kg']} kg, payload {last['payload_kg']} kg "
              f"(peak {peak:.2f}), hole {last['below_kg']} kg, "
              f"particle_max {last['particle_max_m']} m")
        rem, pay = last["removal_kg"], peak
        if rem < 1.0:
            verdict = "the work area was barely touched"
        elif pay < 0.1 * rem:
            verdict = "the layer is being pushed aside, not loaded"
        elif pay > 1.1 * rem:
            verdict = "more soil is held than the work area gave up above the plane"
        else:
            verdict = f"loading {pay / rem:.2f} of what it cut"
        print(f"[play] {verdict}")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
