# scripts/play_dig.py
#
# Run a trained excavate policy with the viewer open, and write a per-step
# trace of the mass balance.
#
#   bash workstation/train.sh -p scripts/play_dig.py
#   bash workstation/train.sh -p scripts/play_dig.py --run 2026-10-24_11-20-38
#   bash workstation/train.sh -p scripts/play_dig.py --no_window --trace dig.csv
#
# Why this exists rather than the CLI's `play`: that subcommand forwards to the
# backend's own script, which does not open the Newton GL viewer this task
# needs to show particles, and does not report where the soil went. The viewer
# block below is dig_demo.py's, unchanged.
#
# --frames writes PNGs if the env can hand back an RGB buffer. On the Newton
# backend it usually cannot, in which case the viewer window is the only
# picture available and the thing to do is screen-record it. The trace is
# written either way and answers most of what the frames would.
#
# What the trace is for: removal, payload and cut_total in kilograms, per step.
# The questions those settle, which eyeballing does not:
#
#   payload tracking removal          the policy is excavating
#   payload flat while removal climbs the layer is being pushed aside
#   payload above removal             soil is coming from under the plane
#   below climbing with advance ~0    the drum is buried and spinning

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)

TASK = "Luna-Excavator-Excavate"
LOG_ROOT = "logs/rsl_rl/excavator_excavate"


def _parse(argv):
    p = argparse.ArgumentParser(description="Play a trained excavate policy.")
    p.add_argument("--task", default=TASK, help=f"default {TASK}")
    p.add_argument("--num_envs", type=int, default=1,
                   help="1 by default: the only count that avoids the multi-world MPM bug")
    p.add_argument("--seconds", type=float, default=25.0)
    p.add_argument("--run", default=None,
                   help=f"run directory under {LOG_ROOT}; default is the newest")
    p.add_argument("--checkpoint", default=None,
                   help="model file inside the run; default is the highest-numbered")
    # Not --headless: AppLauncher guards that name as a SimulationApp config key.
    p.add_argument("--no_window", action="store_true", help="skip the viewer")
    p.add_argument("--park", action="store_true",
                   help="no policy: wheels commanded to zero, arm stowed clear "
                        "of the ground. Isolates whether the machine sinks on "
                        "its own or digs itself in with its wheels")
    p.add_argument("--trace", default="dig_trace.csv", help="per-step CSV")
    p.add_argument("--frames", default=None,
                   help="directory for PNGs, if the env can render to an array. "
                        "The Newton backend hands back nothing, so prefer --dump")
    p.add_argument("--bed_depth", type=float, default=None,
                   help="override bed depth [m]. The sinkage has been 74-92 "
                        "percent of the bed depth in every run regardless of "
                        "material, so if the machine is resting on the floor "
                        "through the soil rather than floating on it, this "
                        "moves and nothing else does")
    p.add_argument("--mpm_voxel", type=float, default=None,
                   help="override the MPM solver grid [m]. The equilibrium "
                        "contact chord sits at about 7 solver cells under "
                        "every load and material tried, so this is the knob "
                        "that should move the sinkage if that is what sets it")
    p.add_argument("--young", type=float, default=None,
                   help="override soil_young_modulus [Pa]. Newton defaults it "
                        "to 1e15, which puts soil-scale elastic strains below "
                        "float32 resolution and leaves the rheology solve "
                        "ill-conditioned")
    p.add_argument("--cohesion", type=float, default=None,
                   help="override soil_cohesion (yield_stress) in Pa. The "
                        "measured capacity scales with gravity, which means "
                        "the cohesion term is contributing nothing; raising "
                        "it tests whether it is connected at all")
    p.add_argument("--gravity_scale", type=float, default=1.0,
                   help="scale gravity, for the load-scaling test. Halving it "
                        "halves the machine's weight; cohesion does not scale "
                        "with it, so a strength-limited sinkage falls roughly "
                        "in proportion while a geometry-limited one does not "
                        "move")
    p.add_argument("--no_rails", dest="rails", action="store_false", default=None,
                   help="put the wheels back in the soil coupling and park the "
                        "rails, which is the configuration that sinks")
    p.add_argument("--rails", dest="rails", action="store_true", default=None,
                   help="wheels on rigid rails at grade (the cfg default)")
    p.add_argument("--probe", action="store_true",
                   help="after stepping, report whether the soil's reaction "
                        "actually reaches the machine's bodies")
    p.add_argument("--dump", default="dig_bed.npz",
                   help="save the soil heightmap per frame here; "
                        "scripts/render_dig.py turns it into a video")
    p.add_argument("--dump_every", type=int, default=4,
                   help="steps between saved frames; 4 is 6.25 fps of sim time")
    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=["newton_gl"])
    args, hydra_args = setup_preset_cli(p, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def _resolve_checkpoint(run: str | None, ckpt: str | None) -> Path:
    root = Path(LOG_ROOT)
    if not root.is_dir():
        raise SystemExit(f"no log root at {root.resolve()}")
    if run is None:
        runs = sorted((d for d in root.iterdir() if d.is_dir()),
                      key=lambda d: d.stat().st_mtime)
        if not runs:
            raise SystemExit(f"no run directories under {root}")
        run_dir = runs[-1]
    else:
        run_dir = root / run
        if not run_dir.is_dir():
            raise SystemExit(f"no run directory {run_dir}")
    if ckpt is not None:
        path = run_dir / ckpt
        if not path.is_file():
            raise SystemExit(f"no checkpoint {path}")
        return path
    models = list(run_dir.glob("model_*.pt"))
    if not models:
        have = sorted(p.name for p in run_dir.iterdir())
        raise SystemExit(f"no model_*.pt in {run_dir}; it holds {have}")
    # model_1199.pt must sort after model_25.pt, so sort on the number.
    def index(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1
    return max(models, key=index)


def _load_policy(device: str, ckpt: Path):
    """Rebuild the actor straight out of the checkpoint.

    Deliberately touches neither rsl_rl nor isaaclab_rl. Going through
    OnPolicyRunner needs the training-time agent config, and the registry copy
    of it does not carry what this rsl_rl build wants: constructing the runner
    from it dies on KeyError 'class_name' inside resolve_class, because Isaac
    Lab fills that in on the train path and load_cfg_from_registry does not.

    Nothing about the file's layout is assumed. rsl_rl has written the weights
    both as one model_state_dict with actor.N.weight keys and as a separate
    actor_state_dict whose keys carry no prefix at all, so the Linears are
    found by looking for 2-D weight tensors, ordered on the last integer in
    the key, and the chain is then checked to connect end to end -- which is
    what would catch a wrong grouping. Widths come from the shapes, so
    actor_hidden_dims is irrelevant here.

    ELU is the one thing no shape can recover, so it is the single assumption,
    and everything found is printed so a mismatch is visible rather than
    silent.
    """
    import torch.nn as nn

    blob = torch.load(str(ckpt), map_location=device, weights_only=False)

    def walk(d, prefix=""):
        """Every tensor in a dict of dicts, keyed by its dotted path."""
        out = {}
        if not isinstance(d, dict):
            return out
        for k, v in d.items():
            path = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(walk(v, f"{path}."))
            elif hasattr(v, "dim"):
                out[path] = v
        return out

    flat = walk(blob)
    if not flat:
        raise SystemExit(f"no tensors anywhere in {ckpt}; it holds {sorted(blob)}")

    # The actor's Linears: 2-D weights that are neither the critic's nor a
    # normaliser's. "actor_state_dict" counts as naming the actor, so a plain
    # "0.weight" under it is kept.
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
    if not mats:
        raise SystemExit(
            f"found no actor Linear weights in {ckpt}.\n"
            f"tensors present: {sorted(flat)}"
        )

    order = sorted(mats)
    print("[play] actor layers found:")
    for i in order:
        key, (out_f, in_f) = mats[i]
        print(f"         {key:40s} {in_f:4d} -> {out_f:4d}")

    # A wrong grouping shows up here, not as garbage actions.
    for a, b in zip(order, order[1:]):
        if mats[a][1][0] != mats[b][1][1]:
            raise SystemExit(
                f"these do not chain: {mats[a][0]} outputs {mats[a][1][0]} but "
                f"{mats[b][0]} takes {mats[b][1][1]}. The layout is not what "
                f"this assumed; tensors present: {sorted(flat)}"
            )

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
            bkey = wkey[: -len("weight")] + "bias"
            if bkey not in flat:
                raise SystemExit(f"{wkey} has no matching {bkey}")
            actor[mine].weight.copy_(flat[wkey])
            actor[mine].bias.copy_(flat[bkey])
    actor.to(device).eval()

    # The normaliser the actor was trained behind, if the run used one.
    # EmpiricalNormalization has been spelt mean/var and _mean/_var, and keeps
    # them shaped (1, D), so match on the trimmed leaf and flatten.
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
        # The mean action, not a sample: this is a readout, not a rollout.
        return actor(x)

    return policy, in_dim


def _find_coupled_solver():
    """The live SolverCoupled, by real type rather than by duck typing.

    The first version of this walked gc for anything answering to
    entry_names/solver/view and found a unittest.mock._Call, which answers to
    every attribute there is. Mocks are why hasattr is not a type test.
    """
    try:
        from isaaclab_contrib.coupling.coupler import NewtonCouplerManager

        solver = getattr(NewtonCouplerManager, "_solver", None)
        if solver is not None:
            return solver
    except Exception:                                      # noqa: BLE001
        pass

    try:
        from newton._src.solvers.coupled.solver_coupled import SolverCoupled
    except Exception as exc:                               # noqa: BLE001
        print(f"[probe] cannot import SolverCoupled: {exc}")
        return None

    import gc

    for obj in gc.get_objects():
        try:
            if isinstance(obj, SolverCoupled):
                return obj
        except Exception:                                  # noqa: BLE001
            continue
    return None


def _probe(u) -> None:
    """Report where the soil's reaction is lost on its way to the machine.

    SolverImplicitMPM.coupling_harvest_proxy_wrenches turns MPM collider
    impulses into proxy-body wrenches, and _harvest_mpm_proxy_wrenches_kernel
    drops a collider silently at any of four tests: an out-of-range collider
    id, a collider that maps to no body, a body that is not a proxy in this
    mapping, and a body without BodyFlags.PROXY. None of them warn. Large
    impulses with no body force means the reaction is generated and then
    discarded at one of those four; impulses near zero means the wheels were
    never registered as MPM colliders at all. Different bugs, different fixes,
    and nothing outside the process tells them apart.
    """
    import numpy as np

    coupled = _find_coupled_solver()
    if coupled is None:
        print("[probe] no coupled solver found")
        return
    print(f"[probe] coupled solver: {type(coupled).__name__}")

    try:
        names = tuple(coupled.entry_names())
    except Exception as exc:                               # noqa: BLE001
        print(f"[probe] entry_names failed: {exc}")
        return
    print(f"[probe] entries: {names}")

    for name in names:
        try:
            solver = coupled.solver(name)
        except Exception as exc:                           # noqa: BLE001
            print(f"[probe] {name}: {exc}")
            continue
        kind = type(solver).__name__
        if not hasattr(solver, "collect_collider_impulses"):
            print(f"[probe] {name}: {kind}")
            continue
        print(f"[probe] {name}: {kind}  <- MPM")

        try:
            state = coupled.entry_state(name, "current")
            imp, pos, ids = solver.collect_collider_impulses(state)
            imp_np = imp.numpy()
            ids_np = ids.numpy()
            mag = np.linalg.norm(imp_np, axis=-1)
            print(f"[probe]   collider impulse entries {ids_np.shape[0]}, "
                  f"valid ids {int((ids_np >= 0).sum())}")
            print(f"[probe]   sum of |impulse| {mag.sum():.4f} N.s "
                  f"(magnitudes, not net)")

            # Net per body. A sum of magnitudes says nothing: impulses are
            # vectors, most of these belong to the ground slab carrying the
            # bed, and around a single wheel they partly cancel. What holds
            # the machine up is the net +z on the eight proxy bodies.
            try:
                import newton

                cbi = solver.collider_body_index.numpy()
                flags = solver.model.body_flags.numpy()
                proxy = int(newton.BodyFlags.PROXY)
                net = np.zeros((flags.shape[0], 3))
                count = np.zeros(flags.shape[0], dtype=int)
                ok = (ids_np >= 0) & (ids_np < cbi.shape[0])
                bodies = cbi[ids_np[ok]]
                good = bodies >= 0
                np.add.at(net, bodies[good], imp_np[ok][good])
                np.add.at(count, bodies[good], 1)
                print("[probe]   net impulse per body (N.s), +z holds it up:")
                for b in range(flags.shape[0]):
                    if count[b] == 0:
                        continue
                    tag = "PROXY" if (flags[b] & proxy) else "kinematic"
                    print(f"[probe]     body {b:2d} {tag:9s} n={count[b]:6d}  "
                          f"net=({net[b][0]:+8.3f},{net[b][1]:+8.3f},{net[b][2]:+9.3f})")
                pm = (flags & proxy) != 0
                zp = net[pm][:, 2].sum()
                print(f"[probe]   proxy bodies net +z: {zp:+.3f} N.s")
                for dt_ in (0.01, 0.04):
                    print(f"[probe]     as force at dt={dt_}: {zp / dt_:+9.1f} N "
                          f"(weight is 797 N)")
            except Exception as exc:                       # noqa: BLE001
                print(f"[probe]   per-body breakdown failed: {exc!r}")
        except Exception as exc:                           # noqa: BLE001
            print(f"[probe]   collect_collider_impulses failed: {exc!r}")

        try:
            cbi = solver.collider_body_index.numpy()
            print(f"[probe]   collider_body_index {cbi.shape[0]} entries, "
                  f"{int((cbi >= 0).sum())} map to a body")
        except Exception as exc:                           # noqa: BLE001
            print(f"[probe]   collider_body_index unavailable: {exc!r}")

        # --- the soil's own stress state -------------------------------
        # particle_elastic_strain is the elastic deformation gradient, identity
        # at rest, so eps = sym(F) - I and sigma follows from linear
        # elasticity. The question this answers: does the bed carry the
        # geostatic pressure of its own weight, rho*g*depth? If it does, the
        # stress field works and the failure is at the yield surface. If the
        # pressure reads near zero, the material never develops stress at all
        # and no yield parameter could matter.
        try:
            st = coupled.entry_state(name, "current")
            F = st.mpm.particle_elastic_strain.numpy()
            q = st.particle_q.numpy()
            E = solver.model.mpm.young_modulus.numpy()
            nu = solver.model.mpm.poisson_ratio.numpy()
            mu = solver.model.mpm.friction.numpy()
            ys = solver.model.mpm.yield_stress.numpy()

            # Raw field first. It is documented as the elastic deformation
            # gradient and is filled with the identity at reset, so a resting
            # bed should sit within ~1e-5 of I. If it does not, the derived
            # stress below is meaningless and this is a history or warm-start
            # variable rather than something to convert.
            diag = np.einsum("...ii->...i", F)
            off = F - np.eye(3) * diag[:, :, None] * 0
            dev_from_I = F - np.eye(3)
            print(f"[probe]   particle_elastic_strain raw: "
                  f"diag mean {diag.mean():.6f} "
                  f"min {diag.min():.4f} max {diag.max():.4f}")
            print(f"[probe]     |F - I| mean {np.abs(dev_from_I).mean():.6f} "
                  f"max {np.abs(dev_from_I).max():.4f}  "
                  f"(a resting bed at E={E.mean():.1g} Pa should be ~1e-5)")

            eps = 0.5 * (F + np.transpose(F, (0, 2, 1))) - np.eye(3)
            tr = np.trace(eps, axis1=1, axis2=2)
            lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
            two_g = E / (1 + nu)
            sig = two_g[:, None, None] * eps + (lam * tr)[:, None, None] * np.eye(3)
            trs = np.trace(sig, axis1=1, axis2=2)
            pres = -trs / 3.0
            dev = sig + (pres)[:, None, None] * np.eye(3)
            tau = np.sqrt(np.sum(dev * dev, axis=(1, 2)))
            limit = ys + (mu / np.sqrt(1.5)) * np.clip(pres, 0.0, None)

            z = q[:, 2]
            top = float(np.percentile(z, 99.5))
            print(f"[probe]   soil stress over {z.shape[0]} particles, "
                  f"surface at z={top:.3f}")
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
            print(f"[probe]   pressure/geostatic tells you whether the bed "
                  f"carries its own weight; tau/yield whether it is at failure")
        except Exception as exc:                           # noqa: BLE001
            print(f"[probe]   soil stress unavailable: {exc!r}")

        try:
            import newton

            flags = solver.model.body_flags.numpy()
            proxy = int(newton.BodyFlags.PROXY)
            kin = int(newton.BodyFlags.KINEMATIC)
            print(f"[probe]   bodies {flags.shape[0]}, "
                  f"PROXY {int((flags & proxy != 0).sum())}, "
                  f"KINEMATIC {int((flags & kin != 0).sum())}")
        except Exception as exc:                           # noqa: BLE001
            print(f"[probe]   body flags unavailable: {exc!r}")


def main(argv=None) -> int:
    args = _parse(argv)
    torch.manual_seed(0)

    ckpt = None
    if args.park:
        print("[play] parked: zero wheel command, arm stowed, no policy")
    else:
        ckpt = _resolve_checkpoint(args.run, args.checkpoint)
        print(f"[play] checkpoint {ckpt}")

    from luna_hifi_tasks.excavator.excavator_cfg import usd_status
    problem = usd_status()
    if problem is not None:
        print(f"\n[asset] {problem}\n")
        return 2

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs

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

    if args.park or args.probe:
        # A reset respawns the machine in the air and re-lays the bed, so any
        # measurement taken after one describes a fresh scene rather than a
        # loaded one. Hold the episode open past the run instead: a parked run
        # has nothing to learn from a reset anyway, and the probe below reads
        # solver state that only means something under load.
        want = args.seconds + 2.0
        if env_cfg.episode_length_s < want:
            print(f"[play] episode_length_s {env_cfg.episode_length_s} -> {want} "
                  f"so no reset lands inside the run")
            env_cfg.episode_length_s = want

    if args.rails is not None:
        env_cfg.wheel_rails = args.rails
        # soil_contact_regex follows wheel_rails in __post_init__, so clearing
        # it here is not required -- but saying what changed is, because the
        # whole point of the run is which surface the wheels are on.
        print(
            f"[play] wheel_rails set to {args.rails}: wheels on "
            f"{'rigid rails' if args.rails else 'soil'}"
        )

    if args.gravity_scale != 1.0:
        g = tuple(float(c) * args.gravity_scale for c in env_cfg.sim.gravity)
        env_cfg.sim.gravity = g
        print(f"[play] gravity scaled by {args.gravity_scale}: {g}")

    if not args.no_window:
        # dig_demo.py's block, unchanged.
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        env_cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                show_particles=True,
                particle_color=(0.62, 0.55, 0.45),
                eye=(5.0, 5.0, 3.0),
                lookat=(0.5, 0.0, 0.3),
            )
        ]

    args.device = env_cfg.sim.device
    env_cfg.validate()

    frame_dir = Path(args.frames) if args.frames else None
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if frame_dir else None)
        u = env.unwrapped
        if args.park:
            # [forward, yaw, boom, drum, shroud]. Boom at -1 is the stow angle,
            # not 0: zero is mid-range, which would drive the arm down into the
            # bed and plant the drums, which is the opposite of a static test.
            # The wheels are a velocity drive, so 0 commands them to hold still
            # rather than to freewheel.
            parked = torch.zeros(u.num_envs, 5, device=u.device)
            parked[:, 2] = -1.0
            policy, in_dim = (lambda _obs: parked), None
        else:
            policy, in_dim = _load_policy(str(u.device), ckpt)

        dt = u.step_dt
        steps = int(args.seconds / dt)
        reset_steps: list[int] = []
        soil = u.cfg.soil_density
        print(f"[play] {steps} steps of {dt:.4f}s, {u.cfg.bed_particles_per_env} particles/env")

        def scalar(name: str, default: float = 0.0) -> float:
            """env attribute for env 0, or `default` where it does not exist.

            A checkpoint is worth replaying against the env it was trained on,
            not against whatever the reward has become since, so this script
            reads whatever that revision happens to expose instead of pinning
            the working tree to one. _cut_total_kg and _escaped arrived with
            the gate on the hole; _removed_kg and _payload_kg one commit
            earlier; none of them exist in the runs before that.
            """
            buf = getattr(u, name, None)
            if buf is None:
                return default
            try:
                return float(buf[0])
            except (TypeError, IndexError):
                return default

        missing = [n for n in ("_removed_kg", "_payload_kg", "_cut_total_kg",
                               "_escaped", "_particle_max_m", "_shape_done",
                               "_below_prev")
                   if getattr(u, n, None) is None]
        if missing:
            print(f"[play] this env revision has no {', '.join(missing)}; "
                  f"those columns are derived or zero")

        # The soil surface itself, which is what a renderer would have been
        # asked to draw. (E, ny, nx) in env-local coordinates, one cell per
        # voxel, recomputed every step by the reward. Saving it costs nothing
        # and is not subject to whether this backend can hand back pixels.
        beds: list = []
        poses: list = []

        rows: list[dict] = []
        obs, _ = env.reset()
        actor_obs = obs["policy"] if isinstance(obs, dict) else obs
        if in_dim is not None and actor_obs.shape[-1] != in_dim:
            raise SystemExit(
                f"the checkpoint's actor takes {in_dim} inputs and this env's "
                f"actor group is {actor_obs.shape[-1]} wide. The observation "
                f"spec changed since the run, so the weights do not apply."
            )
        frames_written = 0
        frames_off = frame_dir is None

        for i in range(steps):
            with torch.inference_mode():
                action = policy(actor_obs).clamp(-1.0, 1.0)
                obs, _, term, trunc, _ = env.step(action)
                actor_obs = obs["policy"] if isinstance(obs, dict) else obs
                dones = term | trunc

            root = u.robot.data.root_pos_w.torch
            below = scalar("_below_prev") * soil
            removal = scalar("_removed_kg")
            rows.append({
                "t": round(i * dt, 4),
                "removal_kg": round(removal, 3),
                "payload_kg": round(scalar("_payload_kg"), 3),
                # Derived when the env predates the field, so one trace format
                # covers every revision of the reward.
                "cut_total_kg": round(scalar("_cut_total_kg", removal + below), 3),
                "below_kg": round(below, 3),
                "advance_mps": round(float(u.robot.data.root_lin_vel_b.torch[0, 0]), 4),
                "chassis_z": round(float(root[0, 2] - u.scene.env_origins[0, 2]), 4),
                "particle_max_m": round(scalar("_particle_max_m"), 2),
                "shape_done": int(scalar("_shape_done")),
                "escaped": int(scalar("_escaped")),
            })

            if args.dump and i % max(1, args.dump_every) == 0:
                grid = getattr(u, "_bed_grid", None)
                if grid is not None:
                    beds.append(grid[0].detach().float().cpu().numpy().copy())
                    origin = u.scene.env_origins[0]
                    poses.append([
                        float(root[0, 0] - origin[0]),
                        float(root[0, 1] - origin[1]),
                        float(root[0, 2] - origin[2]),
                        float(u._base_yaw()[0]) if hasattr(u, "_base_yaw") else 0.0,
                        float(i * dt),
                        removal,
                        scalar("_payload_kg"),
                    ])

            if frame_dir is not None:
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

            if bool(dones.any()):
                reset_steps.append(i)
                print(f"[play] t={i*dt:5.2f}s episode ended "
                      f"(shape_done={int(scalar('_shape_done'))}, "
                      f"escaped={int(scalar('_escaped'))})")

        if args.probe:
            if reset_steps:
                since = steps - 1 - reset_steps[-1]
                print(
                    f"\n[probe] WARNING: the episode reset {since} step(s) before "
                    f"this probe (t={reset_steps[-1]*dt:.2f}s). The machine was "
                    f"respawned clear of the bed and the soil was re-laid, so the "
                    f"impulses and the stress state below describe a fresh scene, "
                    f"not a loaded one. Read chassis_z from the trace instead, and "
                    f"re-run with a longer episode for a probe that means something."
                )
            _probe(u)

        if args.dump and beds:
            import numpy as np

            work = getattr(u, "_work_mask", None)
            np.savez_compressed(
                args.dump,
                bed=np.stack(beds).astype("float32"),
                pose=np.asarray(poses, dtype="float32"),
                work_mask=(work[0].detach().cpu().numpy() if work is not None
                           else np.zeros(beds[0].shape, dtype=bool)),
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

        last = rows[-1]
        peak = max(r["payload_kg"] for r in rows)
        print(f"[play] end: removal {last['removal_kg']} kg, payload {last['payload_kg']} kg "
              f"(peak {peak:.2f}), hole {last['below_kg']} kg, "
              f"particle_max {last['particle_max_m']} m")
        # The one line that says which of the four behaviours this is.
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
