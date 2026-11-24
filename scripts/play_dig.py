# scripts/play_dig.py
#
# Run a trained excavate policy (or a parked or scripted machine) with the
# Newton viewer, and write a per-step CSV trace of the mass balance and a
# heightmap dump for render_dig.py / render_attitude.py. The run ends when the
# episode does, so all of these cover that one episode.
#
#   bash workstation/train.sh -p scripts/play_dig.py
#   bash workstation/train.sh -p scripts/play_dig.py --run 2026-10-24_11-20-38
#   bash workstation/train.sh -p scripts/play_dig.py --no_window --trace dig.csv
#
# Trace columns: removal_kg, payload_kg and cut_total_kg are the env's mass
# balance; below_kg is soil taken from under the plane. The torques are the
# front arm's and front drum's joints and the sum over the wheels, N.m.
# soil_fx_n and soil_fz_n are the soil's force on the front drum (rotor and
# shroud), world frame, and drum_depth_mm its vane tips below the bed surface.

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
import zipfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)
from luna_hifi_tasks.excavator.excavator import (
    ARM_LEN,
    BODY_WHEELS,
    CHASSIS_Z,
    DIG_DRUM_SIGN,
    MAST_TOP_Z,
    ROTOR_TIP_R,
    WHEEL_RADIUS,
)
from luna_hifi_tasks.excavator.excavator_cfg import MAX_DRUM_SPEED, MAX_WHEEL_SPEED, usd_status
from luna_hifi_tasks.excavator.mdp.rewards import body_to_world_xy, world_to_body_xy

TASK = "Luna-Excavator-Excavate"
# The viewer's light: the sun 50 degrees up, behind the machine and on the
# video camera's side (its direction points at the sun), so the machine's
# shadow falls ahead of it and away from the camera and the cut opening behind
# the drum is lit, its far wall facing the camera; plain exposure. Newton's
# default sun is high and behind the camera, its exposure 1.6: flat, washed out.
SUN_AZIMUTH, SUN_ELEVATION, EXPOSURE, SHADOWS_M = 150.0, 50.0, 1.0, 20.0
# The video's camera, high to the side of the pass, looking down about 45 degrees
# onto the drum and the cut opening behind it, the whole machine in view.
VIDEO_EYE, VIDEO_LOOKAT = (0.9, 3.0, 3.4), (1.5, 0.0, 0.4)
# --front: the camera 3.6 m from a point just ahead of the front drum, 30 degrees
# round from straight ahead to the left, 40 degrees up, onto the drum's front, its
# shroud and the soil ahead of it, the machine behind. The top of the frame meets
# the ground 3 m inside a --flat bed's edges; below about 34 degrees up it would
# reach past them. The sun on that side, 70 degrees round and 45 up, so the
# front is lit and the machine's shadow falls behind it.
FRONT_EYE, FRONT_LOOKAT, FRONT_SUN = (4.4, 1.4, 2.8), (2.0, 0.0, 0.45), (70.0, 45.0)
# The RFT soil shades dark with depth down to this, m: a 5 cm cut reads half way.
VIDEO_SHADE_DEPTH = 0.10
# --flat: the bed this wide, m, its edges (a 0.4 m drop to the floor) out of the video.
FLAT_BED_X, FLAT_BED_Y = (-8.0, 12.0), (-10.0, 10.0)
# What a policy is commanded and how its drum-load reading is scaled; --policy_task
# copies these from the policy's task onto the task it runs in.
POLICY_FIELDS = (
    "episode_length_s", "cut_depth_range", "cut_gradient_max", "cut_ramp_fraction", "dig_cut",
    "target_load_bore_fraction",
)


def light(u, azimuth: float = SUN_AZIMUTH, elevation: float = SUN_ELEVATION) -> None:
    """The viewer's light, as site_run's: the sun at azimuth (from +x toward +y)
    and elevation, degrees, its direction pointing at it; EXPOSURE and SHADOWS_M."""
    a, e = math.radians(azimuth), math.radians(elevation)
    for v in getattr(u.sim, "visualizers", None) or []:
        renderer = getattr(getattr(v, "_viewer", None), "renderer", None)
        if renderer is not None:
            renderer.exposure = EXPOSURE
            renderer.shadow_extents = SHADOWS_M
            renderer._sun_direction = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _parse(argv):
    p = argparse.ArgumentParser(description="Play a trained excavate policy.")
    p.add_argument("--task", default=TASK, help=f"default {TASK}")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--seconds", type=float, default=25.0)
    p.add_argument("--run", default=None,
                   help="run directory under the task's logs/rsl_rl/<experiment>; default is the newest")
    p.add_argument("--policy_task", default=None,
                   help="run the newest complete checkpoint of this task instead, with its command, tool "
                        "and drum-load settings: e.g. the RFT policy on MPM soil")
    p.add_argument("--checkpoint", default=None,
                   help="model file inside the run; default is the highest-numbered intact one")
    p.add_argument("--no_window", action="store_true", help="skip the viewer")
    p.add_argument("--park", action="store_true",
                   help="no policy: wheels commanded to zero, arms held at the start angle, drums still")
    p.add_argument("--trace", default="dig_trace.csv", help="per-step CSV")
    p.add_argument("--video", default=None,
                   help="record the run as an mp4 into this directory, from the Newton viewer "
                        "(headless is fine), the camera high to the side of the pass")
    p.add_argument("--flat", action="store_true",
                   help=f"a bed {FLAT_BED_X[1] - FLAT_BED_X[0]:g} x {FLAT_BED_Y[1] - FLAT_BED_Y[0]:g} m, "
                        "its edges out of view: flat ground all round")
    p.add_argument("--front", action="store_true",
                   help=f"the camera ahead of the machine and to its left, at {FRONT_EYE} looking at {FRONT_LOOKAT}, "
                        f"onto the front drum, its shroud and the soil ahead of it; the sun at {FRONT_SUN}")
    p.add_argument("--eye", type=float, nargs=3, default=None, help=f"video camera position; default {VIDEO_EYE}")
    p.add_argument("--lookat", type=float, nargs=3, default=None, help=f"video camera target; default {VIDEO_LOOKAT}")
    p.add_argument("--sun", type=float, nargs=2, default=None, metavar=("AZIMUTH", "ELEVATION"),
                   help=f"the viewer's sun, degrees, azimuth from +x toward +y; "
                        f"default {SUN_AZIMUTH:g} {SUN_ELEVATION:g}, behind the machine on the side camera's side")
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
    p.add_argument("--slope", type=float, default=None,
                   help="this run's cut plane, rise over run along the pass and level across it: "
                        "+ rises (the cut gets shallower), - falls (deeper); the RFT task trains on +-0.10")
    p.add_argument("--depth", type=float, default=None,
                   help="this run's cut depth where the front drum meets soil [m]; the RFT task trains on 0.03 to 0.08")
    p.add_argument("--dig_speed", type=float, default=0.15,
                   help="forward command during --dig, a fraction of MAX_WHEEL_SPEED; negative backs up")
    p.add_argument("--arm_action", type=float, default=0.0,
                   help="front-arm action held during --dig, in [-1, 1]: + lowers the drum at "
                        "dig_arm_rate times it. 0 leaves the arm where the ramp put it")
    # Tool overrides (the env lowers and spins the tool after the settle).
    p.add_argument("--cut", type=float, default=None,
                   help="override dig_cut: vane tips below the surface [m]")
    p.add_argument("--drum", type=float, default=None,
                   help="override dig_drum_speed as a fraction of MAX_DRUM_SPEED; "
                        "the sign of DIG_DRUM_SIGN loads")
    p.add_argument("--inlet_tilt", type=float, default=None,
                   help="override dig_inlet_tilt: inlet angle off straight down [rad]")
    p.add_argument("--intake", type=float, default=None, help="RFT task: override drum_intake [kg/s]")
    p.add_argument("--repose", type=float, default=None,
                   help="RFT task: override soil_repose [deg], which shoves the soil a drum leaves")
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


def _resolve_checkpoint(task: str, run: str | None, ckpt: str | None) -> Path:
    """The named model file in the named or newest run directory of the task's
    experiment; by default the highest-numbered intact model_*.pt, newest run
    first. A checkpoint that a run is still writing, or that a crash cut off,
    is not yet a complete zip archive, and is skipped."""
    root = Path("logs/rsl_rl") / load_cfg_from_registry(task, "rsl_rl_cfg_entry_point").experiment_name
    if run is None:
        runs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    else:
        runs = [root / run]
    if ckpt is not None:
        return runs[0] / ckpt
    for run_dir in runs:
        models = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]), reverse=True)
        for path in models:
            if zipfile.is_zipfile(path):
                return path
            print(f"[play] skipping {path}: not a complete checkpoint")
    raise FileNotFoundError(f"no complete model_*.pt under {root}")


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


def _front_drum_soil_force(u) -> np.ndarray:
    """The soil's force on env 0's front drum (rotor and shroud), world frame, N.

    RFT: the loads rft.py applied. MPM: the soil's contact impulses on the two
    bodies over the last physics step (the MPM entry steps once per physics
    step), over its dt.
    """
    if hasattr(u, "_soil_force"):
        return u._soil_force[0, 0].detach().cpu().numpy()
    from luna_hifi_tasks.excavator.excavate.excavate_env_cfg import MPM_ENTRY

    coupled = u.sim.physics_manager._solver
    solver = coupled.solver(MPM_ENTRY)
    impulse, _, ids = solver.collect_collider_impulses(coupled.entry_state(MPM_ENTRY, "current"))
    impulse, ids = impulse.numpy(), ids.numpy()
    body = solver.collider_body_index.numpy()
    env0 = u.scene.env_prim_paths[0] + "/"
    labels = list(solver.model.body_label)
    front = [b for b, label in enumerate(labels)
             if label.startswith(env0) and label.rsplit("/", 1)[-1] in ("drum_front_body", "shroud_front_body")]
    if len(front) != 2:
        raise RuntimeError(f"front drum and shroud not found among the MPM entry's bodies: {labels}")
    ok = (ids >= 0) & (ids < body.shape[0])
    on_front = np.isin(body[ids[ok]], front)
    return impulse[ok][on_front].sum(axis=0) / u.physics_dt


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
        ckpt = _resolve_checkpoint(args.policy_task or args.task, args.run, args.checkpoint)
        print(f"[play] checkpoint {ckpt}")

    problem = usd_status()
    if problem is not None:
        print(f"\n[asset] {problem}\n")
        return 2

    env_cfg, _ = resolve_task_config(args.task, "")
    if args.policy_task:
        source = load_cfg_from_registry(args.policy_task, "env_cfg_entry_point")
        for field in POLICY_FIELDS:
            setattr(env_cfg, field, getattr(source, field))
        print(f"[play] policy from {args.policy_task}: " + ", ".join(f"{f} {getattr(source, f)}" for f in POLICY_FIELDS))
    env_cfg.scene.num_envs = args.num_envs
    # The sparse grid is sized from max_num_envs; __post_init__ re-derives it below.
    env_cfg.max_num_envs = args.num_envs

    if args.bed_depth is not None:
        env_cfg.bed_depth = args.bed_depth
        print(f"[play] bed_depth set to {args.bed_depth} m")
    if args.flat:
        env_cfg.bed_x, env_cfg.bed_y = FLAT_BED_X, FLAT_BED_Y
        print(f"[play] bed x {FLAT_BED_X}, y {FLAT_BED_Y} m: flat ground all round")

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
                                (args.inlet_tilt, "dig_inlet_tilt", 1.0), (args.dig_ramp, "dig_ramp_seconds", 1.0),
                                (args.intake, "drum_intake", 1.0), (args.repose, "soil_repose", 1.0)):
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

    if not args.no_window or args.video:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        # A video looks down on the pass from high to its side, or with --front onto
        # the front drum from ahead; the window takes in the machine.
        if args.front:
            eye, lookat = FRONT_EYE, FRONT_LOOKAT
        elif args.video:
            eye, lookat = VIDEO_EYE, VIDEO_LOOKAT
        else:
            eye, lookat = (5.0, 5.0, 3.0), (0.5, 0.0, 0.3)
        env_cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                show_particles=True,
                particle_color=(0.62, 0.55, 0.45),
                eye=tuple(args.eye) if args.eye else eye,
                lookat=tuple(args.lookat) if args.lookat else lookat,
            )
        ]
        # The RFT soil has no particles; its env draws the height grid instead.
        if hasattr(env_cfg, "soil_marker"):
            env_cfg.soil_marker = True
    if args.video:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

        # Isaac Lab 3 records through a recorder on the env cfg; render() gives no frames.
        steps = int(args.seconds / (env_cfg.sim.dt * env_cfg.decimation))
        env_cfg.video_recorders = [VideoRecorderCfg(
            source="visualizer:newton", output_dir=args.video, video_length=steps, video_interval=0,
        )]

    if "--device" in sys.argv:
        print(f"[play] sim device set to {args.device} (from --device)")
        env_cfg.sim.device = args.device
    args.device = env_cfg.sim.device
    env_cfg.validate()

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        u = env.unwrapped
        if args.video or not args.no_window:
            light(u, *(args.sun or (FRONT_SUN if args.front else (SUN_AZIMUTH, SUN_ELEVATION))))
            if args.video and hasattr(u, "shade_depth"):
                u.shade_depth = VIDEO_SHADE_DEPTH
        if args.park or args.dig:
            # [drive, steer, front arm rate, front shroud rate]: the drive is
            # dig_forward * a0, and zeros hold the wheels and leave the arm and
            # shroud where the ramp put them.
            fixed = torch.zeros(u.num_envs, u.cfg.action_space, device=u.device)
            if not args.park:
                fixed[:, 0] = args.dig_speed / u.cfg.dig_forward
                fixed[:, 2] = args.arm_action
            if args.dig:
                angle = float(u._dig_arm[0])
                drum = u.cfg.dig_drum_speed
                if drum * DIG_DRUM_SIGN < 0:
                    print(f"[play] *** drums at {drum:+.2f} rad/s is the DUMP "
                          f"direction. Loading is {DIG_DRUM_SIGN:+.0f} ***")
                # Rotor tip height with the chassis at bed_top + CHASSIS_Z + 0.02 (SPAWN_Z).
                chassis0 = float(u.cfg.bed_top) + CHASSIS_Z + 0.02
                tip = chassis0 + MAST_TOP_Z - ARM_LEN * math.sin(angle) - ROTOR_TIP_R
                print(f"[play] DIG: arm {angle:+.4f} rad ({math.degrees(angle):+.1f} deg), "
                      f"shroud {float(u._dig_shroud[0]):+.4f} rad, drums {drum:+.2f} rad/s; rotor tip "
                      f"{1000 * (tip - float(u.cfg.bed_top)):+.0f} mm from the bed surface; "
                      f"forward {args.dig_speed * MAX_WHEEL_SPEED * WHEEL_RADIUS:.2f} m/s "
                      f"and arm {args.arm_action * u.cfg.dig_arm_rate:+.3f} rad/s "
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
        end_t = None   # when the episode ended; the run stops there
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
        wheel_bodies = u.robot.find_bodies(BODY_WHEELS, preserve_order=True)[0]
        obs, _ = env.reset()
        actor_obs = obs["policy"]
        # This episode's cut command. Set here, before the settle places the plane
        # under the surface; the machine does not move before the ramp is done.
        if args.slope is not None:
            grad = torch.tensor([[args.slope, 0.0]], device=u.device).expand(u.num_envs, 2)
            u._target_grad[:] = body_to_world_xy(grad, u._work_yaw)
        if args.depth is not None:
            u._target_depth[:] = args.depth
        g = world_to_body_xy(u._target_grad, u._work_yaw)[0]
        print(f"[play] cut command: {100 * scalar('_target_depth'):.1f} cm deep where the drum meets soil, "
              f"slope {float(g[0]):+.3f} along the pass, {float(g[1]):+.3f} across")
        # This episode's work area; a reset marks a new one.
        work_mask = u._work_mask[0].detach().cpu().numpy()

        for i in range(steps):
            # Read before the step: the reset on the step a pass ends zeroes it.
            fill_prev = u._fill_kg[0].tolist()
            with torch.inference_mode():
                action = policy(actor_obs).clamp(-1.0, 1.0)
                obs, _, term, trunc, extras = env.step(action)
                actor_obs = obs["policy"]
                dones = term | trunc

            if bool(dones[0]):
                # The step that ends the episode also resets the env, so the run stops
                # here: the trace, frames and summary cover the episode that ran. The
                # reset cleared the env's flags; its exit log has them.
                end_t = i * dt
                log = extras["log"]
                print(f"[play] t={end_t:5.2f}s episode ended "
                      f"(pass_done={log['Episode/pass_done']:.0f}, "
                      f"off_bed={log['Episode/off_bed']:.0f}, "
                      f"failed={log['Episode/failed']:.0f}, "
                      f"shape_done={log['Episode/shape_done']:.0f}, "
                      f"escaped={log['Episode/escaped']:.0f}), "
                      f"in the drums: front {fill_prev[0]:.1f} kg, rear {fill_prev[1]:.1f} kg")
                break

            root = u.robot.data.root_pos_w.torch
            tau = u.robot.data.applied_torque.torch[0]
            soil_f = _front_drum_soil_force(u)
            drum_z = float(u.robot.data.body_link_pos_w.torch[0, u._drum_body_ids[0], 2] - u.scene.env_origins[0, 2])
            wheel_z = u.robot.data.body_link_pos_w.torch[0, wheel_bodies, 2] - u.scene.env_origins[0, 2]
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
                "arm_torque_nm": round(float(tau[u._arm_ids[0]]), 1),
                "drum_torque_nm": round(float(tau[u._drum_ids[0]]), 1),
                "wheel_torque_nm": round(float(tau[u._wheel_ids].sum()), 1),
                "soil_fx_n": round(float(soil_f[0]), 1),
                "soil_fz_n": round(float(soil_f[2]), 1),
                "drum_depth_mm": round(1000.0 * (u.cfg.bed_top - (drum_z - ROTOR_TIP_R)), 1),
                # Tyre bottoms below the bed surface, and the tyres' rim speed; mean of the four.
                "wheel_sink_mm": round(1000.0 * float((u.cfg.bed_top - (wheel_z - WHEEL_RADIUS)).mean()), 1),
                "wheel_speed_mps": round(float(u.robot.data.joint_vel.torch[0, u._wheel_ids].mean()) * WHEEL_RADIUS, 4),
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

            if (i + 1) % 100 == 0:
                now = time.perf_counter()
                sps = 100.0 / (now - t_mark)
                t_mark = now
                print(f"[play] step {i + 1}/{steps}  {sps:6.2f} env-steps/s  "
                      f"eta {(steps - i - 1) / sps:6.0f}s", flush=True)

        if args.probe:
            _attitude(u)
            if end_t is not None:
                print(f"\n[probe] WARNING: the episode ended at t={end_t:.2f}s; the readings "
                      f"below describe a freshly reset scene.")
            _probe(u)

        if args.dump and beds:
            np.savez_compressed(
                args.dump,
                bed=np.stack(beds).astype("float32"),
                pose=np.asarray(poses, dtype="float32"),
                work_mask=work_mask,
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

        wall = time.perf_counter() - t_start
        ran = i + 1
        sps = ran / wall
        print(f"[play] {ran} steps in {wall:.1f}s = {sps:.2f} env-steps/s "
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
        moving = rows[u._settle_steps + u._ramp_steps:] or rows
        keys = ("arm_torque_nm", "drum_torque_nm", "wheel_torque_nm", "soil_fx_n", "soil_fz_n", "drum_depth_mm", "advance_mps")
        mean = {k: sum(r[k] for r in moving) / len(moving) for k in keys}
        print(f"[play] torque after the ramp, mean: front arm {mean['arm_torque_nm']:+.1f}, "
              f"front drum {mean['drum_torque_nm']:+.1f}, wheels {mean['wheel_torque_nm']:+.1f} N.m")
        print(f"[play] front drum after the ramp, mean: soil force x {mean['soil_fx_n']:+.1f} N, "
              f"z {mean['soil_fz_n']:+.1f} N; vane tips {mean['drum_depth_mm']:.1f} mm deep; "
              f"advance {mean['advance_mps']:.3f} m/s")
        # Slip: 1 - advance / rim speed, over the steps the wheels turn.
        turning = [r for r in moving if abs(r["wheel_speed_mps"]) > 0.05]
        slip = sum(1.0 - r["advance_mps"] / r["wheel_speed_mps"] for r in turning) / max(len(turning), 1)
        print(f"[play] wheels after the ramp, mean: tyres {sum(r['wheel_sink_mm'] for r in moving) / len(moving):.1f} mm "
              f"below the bed surface, slip {slip:+.2f} over the {len(turning)} steps they turn")
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
        if args.video:
            clips = sorted(Path(args.video).glob("*.mp4"), key=lambda q: q.stat().st_mtime)
            print(f"[play] video: {clips[-1] if clips else 'none written to ' + args.video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
