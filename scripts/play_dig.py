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
import math
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
    p.add_argument("--wheel_mass_scale", type=float, default=None,
                   help="the effective mass each wheel presents to the MPM "
                        "solve, as a multiple of its 14.1 kg free-body mass. "
                        "The wheel carries about 123 kg of the machine, so the "
                        "default is 8.7; 1.0 hands the soil a free wheel and "
                        "is why it sank")
    p.add_argument("--proxy_relaxation", type=float, default=None,
                   help="under-relaxation on the rigid<->MPM force exchange. "
                        "The exchange is a fixed-point iteration of gain "
                        "g = m_proxy/M, so stability wants omega < 2/(1+g) "
                        "and the error per pass is smallest at 1/(1+g); at "
                        "mass scale 8.7 that is 0.21 and 0.10. Default 0.1. "
                        "Sweep this, not the mass: if settled sinkage moves "
                        "with omega the loop is not converging")
    p.add_argument("--coupler_iterations", type=int, default=None,
                   help="rigid<->MPM passes per physics step, default 4. Each "
                        "pass runs both solvers, so this multiplies step cost "
                        "directly: 4 -> 2 is a 2x speedup. It was 4 because "
                        "the exchange ran undamped and needed the passes; with "
                        "proxy_relaxation at its optimum the loop should "
                        "converge in fewer. Verify the same way as omega -- if "
                        "settled sinkage moves, it is too few")
    p.add_argument("--proxy_relaxation_mode", default=None,
                   choices=("fixed", "aitken"),
                   help="'fixed' holds --proxy_relaxation, 'aitken' updates it "
                        "from successive residuals (Irons-Tuck). Keep 'fixed' "
                        "while sweeping omega")
    p.add_argument("--dig", action="store_true",
                   help="scripted dig instead of a policy: lower the arm until "
                        "the shroud sits --cut under the surface, spin the "
                        "drums up, then creep forward. For seeing whether the "
                        "machine can excavate at all, without a checkpoint")
    p.add_argument("--straight", action="store_true",
                   help="zero every spawn randomisation so the machine starts "
                        "on the bed centreline pointed down +x with a level "
                        "target plane. The defaults jitter yaw by +-0.40 rad "
                        "(23 deg), x by +-0.55 m and y by +-0.30 m, which is "
                        "training diversity and pure noise in a single run")
    p.add_argument("--boom", type=float, default=None,
                   help="boom command in [-1, 1], overriding --cut. -1 is the "
                        "stow angle and +1 is full dig. Use this to bypass "
                        "boom_command_for_cut when the question is whether the "
                        "drums can reach the soil at all")
    p.add_argument("--cut", type=float, default=0.10,
                   help="cut depth in m for --dig, measured to the SHROUD. "
                        "Default 0.10 puts the rotor tip about 53 mm into the "
                        "soil. The task's own cut_depth_range is 0.04, which "
                        "leaves the rotor ABOVE the bed and cuts nothing -- "
                        "see the geometry note in the --dig branch")
    p.add_argument("--dig_speed", type=float, default=0.15,
                   help="forward command during --dig, a fraction of "
                        "MAX_WHEEL_SPEED (5 rad/s, 1.5 m/s). Default 0.15 is "
                        "0.22 m/s, which crosses the 5.8 m bed in 26 s")
    p.add_argument("--dig_ramp", type=float, default=3.0,
                   help="seconds to lower the arm from stow to the cut depth "
                        "and spin the drums up, before the machine moves off")
    p.add_argument("--limp", action="store_true",
                   help="zero every actuator gain, so NOTHING is driven. --park "
                        "still commands all four controllers (wheels braked to "
                        "zero velocity, arm held at stow on stiffness 2500, "
                        "drums and shrouds held), which is not the same as "
                        "dropping an unpowered machine on the soil")
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


def _wheel_radius() -> float:
    """WHEEL_RADIUS, read lazily so this script keeps working if the geometry
    module moves. 0.30 is the value it has had throughout."""
    try:
        from luna_hifi_tasks.excavator.excavator import WHEEL_RADIUS  # noqa: PLC0415

        return float(WHEEL_RADIUS)
    except Exception:  # noqa: BLE001
        return 0.30


def _attitude(u) -> None:
    """Commanded vs achieved arm angles, and the machine's actual attitude.

    The parked run that prompted this showed the four wheels 344 mm apart in z,
    front pair clear of the soil and rear pair on the bed floor: the machine was
    not sinking, it was pitched nose-up and standing on its rear wheels. The
    cause is in the same table -- the two arms end 620 mm apart in z at the same
    commanded stow angle, front boom up 30.2 deg and rear boom down 18.2 deg.
    excavator.py builds the rear assembly with _arm_assembly("rear", math.pi,
    ...), a 180 deg yaw, and its own joint-sign note says positive pitches the
    boom DOWN while documenting the yaw only for the drums.

    So print the joint angles beside the body heights instead of inferring the
    convention from geometry.
    """
    try:
        import numpy as np

        d = u.robot.data
        names = list(getattr(d, "joint_names", []) or [])
        q = d.joint_pos.torch[0].detach().cpu().numpy()
        tgt = None
        for attr in ("joint_pos_target", "joint_pos_target_w"):
            t = getattr(d, attr, None)
            if t is not None:
                tgt = t.torch[0].detach().cpu().numpy()
                break
        print("[attitude] arm and shroud joints, commanded vs achieved:")
        for i, n in enumerate(names):
            if not ("arm" in n or "shroud" in n):
                continue
            c = f"{tgt[i]:+8.4f}" if tgt is not None and i < len(tgt) else "       ?"
            print(f"[attitude]   {n:14s} q={q[i]:+8.4f} rad ({np.degrees(q[i]):+7.2f} deg)"
                  f"  target={c}")

        bl = list(getattr(u.robot.data, "body_names", []) or [])
        bz = d.body_pos_w.torch[0, :, 2].detach().cpu().numpy()
        want = [i for i, n in enumerate(bl) if "wheel" in n or "drum" in n]
        # A drum's body z is its AXIS. Whether it is in the soil is decided by
        # its lowest point, which is a rotor tip radius below that, and the
        # difference is 18.5 cm -- enough to read a drum hanging clear of the
        # bed as a drum that is cutting. So report the bottom and the gap.
        try:
            from luna_hifi_tasks.excavator.excavator import (  # noqa: PLC0415
                ROTOR_TIP_R,
                WHEEL_RADIUS,
            )
        except Exception:  # noqa: BLE001
            ROTOR_TIP_R, WHEEL_RADIUS = 0.185, 0.30
        surf = float(getattr(u.cfg, "bed_top", float("nan")))
        if want:
            print(f"[attitude] body heights (world z), bed surface z={surf:+.4f}:")
            for i in want:
                r = WHEEL_RADIUS if "wheel" in bl[i] else ROTOR_TIP_R
                bot = bz[i] - r
                print(f"[attitude]   {bl[i]:22s} axis z={bz[i]:+8.4f}  "
                      f"bottom={bot:+8.4f}  {1000 * (bot - surf):+7.0f} mm "
                      f"{'BELOW' if bot < surf else 'above'} the bed")
            wz = [bz[i] for i in want if "wheel" in bl[i]]
            if len(wz) >= 2:
                print(f"[attitude]   wheel z spread {1000 * (max(wz) - min(wz)):.0f} mm "
                      f"-- a level machine on flat soil should be a few mm")
    except Exception as exc:  # noqa: BLE001
        print(f"[attitude] unavailable: {exc!r}")


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
                    # Every body, including n=0. Hiding the empty rows is what
                    # made this table unreadable: with bodies 0-3 dropped, the
                    # four that remained looked like "the four wheels, two of
                    # them starved", when 0-3 were the stowed drums and shrouds
                    # and 4-7 were the wheels. Local indices here are the MPM
                    # entry's compacted ModelView, i.e. sorted global ids over
                    # the bodies that entry owns plus the proxies -- not global
                    # body indices, and not the order anything else prints.
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

        # --- who is body b, and is it even in the race -----------------
        #
        # Three things make the per-body node count unreadable on its own,
        # and this block measures all three.
        #
        # 1. These body indices are LOCAL to the MPM entry's compacted
        #    ModelView, not the articulation's. SolverCoupled builds that
        #    view from sorted(global ids) over the bodies the entry owns
        #    plus the proxy bodies (solver_coupled.py _ordered_world_subset
        #    returns sorted(indices) for a single world), so local 0 is just
        #    the lowest global id in that set and nothing about it lines up
        #    with u.robot.body_names. body_label survives the compaction;
        #    print it instead of inferring the order from the MJCF.
        #
        # 2. Attribution is winner-take-all, not a sum. ONE mesh is built
        #    per BODY -- implicit_mpm_model.py _create_body_collider_mesh
        #    merges every COLLIDE_PARTICLES shape of a body into a single
        #    wp.Mesh -- and collision_sdf (rasterized_collisions.py:265-278)
        #    keeps only the collider with the smallest sdf at each node. So
        #    n is "nodes this body WON", and a body loses a node to any
        #    other body that is nearer. The 119 geoms do not compete; the
        #    13 bodies do.
        #
        # 3. The comparison is on sdf = distance - thickness
        #    (rasterized_collisions.py:220), and thickness is the per-SHAPE
        #    shape_margin. The bed slab and the retaining plates are spawned
        #    with contact_margin = MPM_COLLIDER_MARGIN = 0.02; the excavator
        #    comes from a USD with no margin authored, so its geoms get
        #    ShapeConfig.margin = 0.0. A body with the larger margin is
        #    inflated and wins nodes it is not nearest to, and its
        #    activation band reaches margin + activation instead of
        #    activation. If the machine reads 0.000 and the slab 0.020, a
        #    wheel resting on the slab cannot win a node over its own
        #    contact patch, and that is the 1-vs-345 split.
        try:
            import newton

            cbi = solver.collider_body_index.numpy()
            print(f"[probe]   collider_body_index {cbi.shape[0]} entries, "
                  f"{int((cbi >= 0).sum())} map to a body")

            mdl = solver.model
            flags = mdl.body_flags.numpy()
            nb = int(flags.shape[0])
            proxy = int(newton.BodyFlags.PROXY)
            kinflag = int(newton.BodyFlags.KINEMATIC)

            labels = ["?"] * nb
            for attr in ("body_label", "body_key"):
                raw = getattr(mdl, attr, None)
                if raw is not None and len(raw) >= nb:
                    labels = [str(raw[b]) for b in range(nb)]
                    print(f"[probe]   body names from model.{attr}")
                    break
            else:
                print("[probe]   no body_label/body_key on the view")

            margin = mdl.shape_margin.numpy()
            sflags = mdl.shape_flags.numpy()
            scale = mdl.shape_scale.numpy()
            sxf = mdl.shape_transform.numpy()
            bshapes = getattr(mdl, "body_shapes", {}) or {}
            collide = int(newton.ShapeFlags.COLLIDE_PARTICLES)

            vox = float(solver._mpm_model.voxel_size)
            try:
                deg = int(solver._scratchpad.collider_position_field.degree)
            except Exception:                              # noqa: BLE001
                deg = -1
            act = 0.0 if deg == 0 else (0.5 / deg) * vox
            print(f"[probe]   voxel {vox:.3f} m, collider basis "
                  f"{getattr(solver, 'collider_basis', '?')} degree {deg}, "
                  f"activation {act:.4f} m (a pic* basis is degree 0 -> 0.0, "
                  f"so the machine gets no band and only margin counts)")

            bpos = None
            try:
                bq = state.body_q.numpy()
                if bq.shape[0] == nb:
                    bpos = bq[:, :3]
            except Exception:                              # noqa: BLE001
                pass

            # Per-body bounding radius in body-local coords, over the shapes
            # that actually became collider geometry. Loose over-bound:
            # |shape offset| + largest half-extent.
            radius = np.zeros(nb)
            ngeom = np.zeros(nb, dtype=int)
            shape_ids = {}
            for b in range(nb):
                sl = [int(s) for s in bshapes.get(b, ())
                      if sflags[int(s)] & collide]
                shape_ids[b] = sl
                ngeom[b] = len(sl)
                for s in sl:
                    off = float(np.linalg.norm(sxf[s][:3]))
                    radius[b] = max(radius[b],
                                    off + float(np.max(np.abs(scale[s]))))

            print("[probe]   per-body attribution; n is nodes WON, not force:")
            print("[probe]      b tag       geoms       margin   reach "
                  "     n    sum|f|     |net|   maxnode        z  name")
            reach = np.zeros(nb)
            for b in range(nb):
                mg = margin[shape_ids[b]] if shape_ids[b] else np.zeros(1)
                reach[b] = float(mg.max()) + act
                if count[b]:
                    fb = imp_np[ok][good][bodies[good] == b]
                    fm = np.linalg.norm(fb, axis=-1)
                    smag, nmag = float(fm.sum()), float(
                        np.linalg.norm(fb.sum(axis=0)))
                    mx = float(fm.max())
                else:
                    smag = nmag = mx = 0.0
                tag = ("PROXY" if flags[b] & proxy else
                       "KINEMATIC" if flags[b] & kinflag else "plain")
                zs = f"{bpos[b][2]:+8.4f}" if bpos is not None else "       ?"
                print(f"[probe]     {b:2d} {tag:9s} {ngeom[b]:5d} "
                      f"{mg.min():.3f}-{mg.max():.3f}  {reach[b]:.4f} "
                      f"{count[b]:6d} {smag:9.3f} {nmag:9.3f} {mx:9.3f} "
                      f"{zs}  {labels[b]}")
            print("[probe]     n=0 is either clear of the soil (read z) or "
                  "every node lost to a body with a longer reach")

            # --- nodes lost to the argmin race -------------------------
            # Per (loser, winner): active nodes inside the loser's own reach
            # that were attributed to the winner. This is the number n
            # hides. The test is against a bounding sphere rather than the
            # mesh, so read the count as "at most this many were contested";
            # the winner column is the part that matters.
            if bpos is not None:
                apos = pos.numpy()[ok][good]
                awin = bodies[good]
                print("[probe]   contested nodes, inside one body's reach "
                      "but won by another:")
                for b in range(nb):
                    if ngeom[b] == 0:
                        continue
                    near = (np.linalg.norm(apos - bpos[b], axis=1)
                            <= radius[b] + reach[b])
                    lost = near & (awin != b)
                    if not lost.any():
                        continue
                    w, c = np.unique(awin[lost], return_counts=True)
                    top = ", ".join(
                        f"{labels[int(wi)].rsplit('/', 1)[-1]}({int(ci)})"
                        for wi, ci in sorted(zip(w, c), key=lambda t: -t[1])[:4]
                    )
                    print(f"[probe]     {labels[b].rsplit('/', 1)[-1]:24s} "
                          f"kept {count[b]:6d}  lost {int(lost.sum()):6d}  "
                          f"to {top}")
        except Exception as exc:                           # noqa: BLE001
            print(f"[probe]   attribution breakdown failed: {exc!r}")

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

            # particle_stress is the solver's OWN yield-projected stress in
            # Pa (registered as "mpm:particle_stress", written from
            # project_stress). Read it: it is the operative stress and it is on
            # the force path. Everything derived from F below is a diagnostic.
            #
            # SIGN: it is COMPRESSION-POSITIVE, i.e. -sigma_Cauchy, and the
            # attribute doc at solver_implicit_mpm.py:1222 ("Cauchy stress
            # tensor [Pa]") is sign-wrong. Proof from source, no measurement
            # needed. The field is dof_to_value_3d(project_stress(...)) and
            # warp.fem SymmetricTensorMapper.value_to_dof_3d sets
            # r[0] = trace/sqrt(6) = sqrt(1.5) * trace/3, while project_stress
            # clamps r[0] into [-tensile_ratio*yield_pressure*sqrt(1.5),
            # +yield_pressure*sqrt(1.5)] -- a POSITIVE upper bound named
            # p_max, the COMPRESSIVE yield pressure. So +trace/3 is
            # compressive pressure, and -trace/3 had the sign backwards.
            #
            # INVARIANTS: the mapper is orthonormal for the half-Frobenius
            # norm (tau:tau)/2, so the solver's |r_T| is
            # sqrt(dev:dev / 2) = sqrt(J2) exactly -- NOT ||dev||_F, which is
            # sqrt(2) larger. And shear_yield_stress returns
            # s + mu*(r_N - p_min) with mu = friction/sqrt(1.5) applied to
            # r_N = sqrt(1.5)*p, so the sqrt(1.5) CANCELS and the yield
            # surface is plain Mohr-Coulomb in the physical pressure:
            #
            #     sqrt(J2) <= yield_stress + friction * p
            #
            # Dividing friction by sqrt(1.5) and then applying it to p
            # understates the limit by up to 19% at these pressures.
            try:
                sg = st.mpm.particle_stress.numpy()
                pp = np.trace(sg, axis1=1, axis2=2) / 3.0   # compression +
                dv = sg - pp[:, None, None] * np.eye(3)
                tt = np.sqrt(0.5 * np.einsum("nij,nij->n", dv, dv))
                lim = ys + mu * np.clip(pp, 0.0, None)
                rat = tt / np.maximum(lim, 1e-9)
                # project_stress clamps r_N at p_min, and p_min is 0 whenever
                # tensile_yield_ratio is 0 -- Newton's default, which this cfg
                # never overrides. So p >= 0 is a HARD FLOOR, not a reading.
                # Count how much of the bed sits on it and report the LOW tail
                # of p: a p99 of the sign-flipped pressure is the low tail
                # dressed up as the high one, which is how "p99 = 0 Pa" came
                # to be read as "most of the bed carries no pressure".
                npart = int(pp.shape[0])
                floor = int((pp <= 0.0).sum())
                print(f"[probe]   particle_stress (compression-POSITIVE): "
                      f"p mean {pp.mean():.1f}Pa "
                      f"p1 {np.percentile(pp, 1):.1f}Pa "
                      f"p50 {np.percentile(pp, 50):.1f}Pa "
                      f"p99 {np.percentile(pp, 99):.1f}Pa "
                      f"(expect p ~ rho*g*depth, 243..2430 Pa)")
                print(f"[probe]     sqrt(J2) mean {tt.mean():.1f}Pa, "
                      f"yield = {ys.mean():.0f} + {mu.mean():.3f}*p, "
                      f"tau/yield mean {rat.mean():.3f} "
                      f"p99 {np.percentile(rat, 99):.3f}  "
                      f"(1.0 is failure; the solver projects onto the surface, "
                      f"so >1 is impossible and means the arithmetic is wrong)")
                # Two different defects both land a particle at p = 0, and
                # sqrt(J2) tells them apart:
                #   p == 0 and sqrt(J2) > 0  -> project_stress clamped the
                #     trace at p_min and kept the deviator. Legitimate for a
                #     free-surface or dilating particle.
                #   sigma == 0 identically   -> the particle has NO strain
                #     node. strain_basis "pic8" is
                #     PointBasisSpace(pic, max_nodes_per_element=8), and
                #     warp fem UnstructuredPointTopology.element_node_count
                #     (space/point_basis_space.py:124) returns
                #     min(8, particles_in_cell) with no warning. This bed
                #     packs exactly 8 particles into 5184 of its 7592 cells
                #     at spawn, so the cap has ZERO headroom: ~8% of vertical
                #     compaction drops 13.6% of the bed out of the strain
                #     space, and the 42% compaction a 106 mm rut implies in a
                #     250 mm bed drops 41%. A particle with no strain node
                #     carries no stress and obeys no yield criterion.
                dead = int(((np.abs(pp) <= 0.0) & (tt <= 0.0)).sum())
                print(f"[probe]     p <= 0: {floor} of {npart} particles "
                      f"= {100.0 * floor / max(npart, 1):.2f}%  "
                      f"(the project_stress clamp at p_min=0, which is what "
                      f"tensile_yield_ratio=0 gives, makes p >= 0 a hard "
                      f"floor, so this is the LOW tail of p, not the high one)")
                print(f"[probe]     sigma == 0 identically: {dead} of {npart} "
                      f"= {100.0 * dead / max(npart, 1):.2f}%  "
                      f"(these have no strain node at all -- the pic8 cap in "
                      f"warp fem point_basis_space.py:124 -- and are carrying "
                      f"and resisting nothing)")

                # Pressure UNDER the loaded wheels against pressure away from
                # them, at the same depth. This decides whether the frictional
                # half of Drucker-Prager does anything at all. Strength is
                # cohesion + friction*p, so if p beneath a loaded footing is no
                # higher than the geostatic p elsewhere, friction contributes
                # nothing and the soil can only carry its unconfined
                # compressive strength: 500/(1/sqrt(3) - 0.84/3) = 1682 Pa.
                # A wheel at 399 N held by that alone needs 0.237 m2, which is
                # 30 cm of sinkage. Measured 23 cm, where it ran out of bed.
                # Terzaghi with confinement gives 151 kPa at 1 cm, a factor of
                # 90, and Apollo measured 1-3 cm of rut.
                try:
                    bp = u.robot.data.body_pos_w.torch[0].detach().cpu().numpy()
                    bn = list(getattr(u.robot.data, "body_names", []) or [])
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
                        print("[probe]   pressure UNDER the loaded wheels vs away, "
                              "same depth:")
                        print(f"[probe]     loaded: "
                              f"{[bn[k].replace('_body', '') for k in low]}")
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
                        print("[probe]     ratio ~1 means the load raises no extra "
                              "confining pressure, so")
                        print("[probe]     friction*p adds nothing and cohesion alone "
                              "carries the wheel")
                except Exception as exc:  # noqa: BLE001
                    print(f"[probe]   under-wheel pressure unavailable: {exc!r}")
            except Exception as exc:                       # noqa: BLE001
                print(f"[probe]   particle_stress unavailable: {exc!r}")

            # DO NOT use sym(F) - I. F is the elastic DEFORMATION GRADIENT and
            # it carries accumulated rigid rotation: the particle update adds
            # skew = 0.5*dt*(L - L^T) to the strain increment, so F = R(I + e)
            # with |e| ~ 1e-5 and R an accumulated rotation of tens of degrees
            # after a churning bed. Symmetrising a rotation by angle th about
            # ANY axis n gives sym(R) - I = (cos th - 1)(I - n (x) n), an
            # axisymmetric *compressive* strain whose probe-derived invariants
            # are tau/p = (sqrt(6)/3)*G/K = 0.37684 and
            # tau/yield = that * sqrt(1.5)/friction = 0.549451 -- independent
            # of th, of the axis, of depth and of load. That is exactly the
            # "7 MPa, flat with depth, tau/yield pinned at 0.549" artifact, and
            # max|F - I| -> 2 is the geometric cap of a rotation, not strain.
            # The solver itself is rotation-correct: strain_rhs SVDs F and
            # takes U log(Sigma) U^T, discarding R. Do the same here.
            Fu, Fs, _Fv = np.linalg.svd(F)
            eps = np.einsum("nik,nk,njk->nij", Fu,
                            np.log(np.maximum(Fs, 1e-12)), Fu)
            tr = np.trace(eps, axis1=1, axis2=2)
            lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
            two_g = E / (1 + nu)
            sig = two_g[:, None, None] * eps + (lam * tr)[:, None, None] * np.eye(3)
            trs = np.trace(sig, axis1=1, axis2=2)
            # Compression-POSITIVE, to match particle_stress above. sig here
            # is built from the elastic law, so its trace is negative in
            # compression and -trace/3 is the right sign for THIS tensor --
            # the opposite of particle_stress. Two conventions in one function
            # is what made the two readings look irreconcilable.
            pres = -trs / 3.0
            dev = sig - (trs / 3.0)[:, None, None] * np.eye(3)
            # sqrt(J2), to match the solver's |r_T|. np.sum(dev*dev) is
            # ||dev||_F^2 = 2*J2, so the bare sqrt of it was sqrt(2) too big.
            tau = np.sqrt(0.5 * np.einsum("nij,nij->n", dev, dev))
            # yield = cohesion + friction*p. No sqrt(1.5): it cancels against
            # r_N = sqrt(1.5)*p. See the note on INVARIANTS above.
            limit = ys + mu * np.clip(pres, 0.0, None)

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

    # The sparse-grid capacities are absolute totals derived from max_num_envs,
    # NOT from scene.num_envs, so a one-environment run otherwise allocates the
    # grid for all 104: 50,750 particles x 104 = 5.28M, active = 2^23, about
    # 25,770 MB against a 3090's 24,576 MB. That fails as
    # "CUDA error 2: out of memory" in wp_alloc_device_async and
    # rebuild_count_points, followed by a storm of illegal-access errors from
    # the failed allocation -- which reads exactly like a solver bug and is not
    # one. __post_init__ is what derives the capacities, so it has to run again.
    # scripts/run_task.py has carried this since ed539c6; this script never did.
    if hasattr(env_cfg, "max_num_envs"):
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

    if args.park or args.probe or args.dig:
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

    if args.straight:
        # Spawn randomisation is for training diversity. In a single run it
        # only rotates the plot and drives the machine at a wall.
        for field, zero in (
            ("spawn_yaw_jitter", 0.0),
            ("spawn_x_jitter", 0.0),
            ("spawn_y_jitter", 0.0),
            ("arm_start_jitter", 0.0),
            ("cut_gradient_max", 0.0),
        ):
            if hasattr(env_cfg, field):
                setattr(env_cfg, field, zero)
        print("[play] STRAIGHT: yaw, xy, arm and gradient jitter all zeroed; "
              "machine spawns on the centreline pointed down +x")

    if args.coupler_iterations is not None:
        env_cfg.coupler_iterations = args.coupler_iterations
        print(f"[play] coupler_iterations set to {args.coupler_iterations}")

    if args.proxy_relaxation is not None:
        env_cfg.proxy_relaxation = args.proxy_relaxation
        g = (args.wheel_mass_scale
             if args.wheel_mass_scale is not None
             else env_cfg.wheel_proxy_mass_scale)
        print(
            f"[play] proxy_relaxation set to {args.proxy_relaxation:g} "
            f"(gain g~{g:g} wants omega < {2.0 / (1.0 + g):.3f}, "
            f"best near {1.0 / (1.0 + g):.3f})"
        )

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

    # Every override above feeds a derivation in __post_init__ -- the grid
    # capacities from max_num_envs, soil_contact_regex and the proxy mappings
    # from wheel_rails, the rail geometry from bed_depth -- so it has to run
    # after them. It used to run immediately after max_num_envs, which left
    # --no_rails and the rest writing fields that nothing re-read.
    if hasattr(env_cfg, "max_num_envs"):
        env_cfg.__post_init__()

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

    # add_launcher_args already defines --device, so this script must not. It
    # also must not silently discard it: the line below used to overwrite
    # args.device with the cfg's value, so passing --device did nothing.
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
        if args.park:
            # NOTE: this is not an unactuated machine. Every one of these four
            # controllers is live and holding a setpoint -- the wheels are a
            # velocity drive commanded to 0, so they are BRAKED rather than
            # free to roll, and the arm actuator holds ~46 kg of drum and
            # shroud on a 0.76 m lever at stiffness 2500, about 48 N.m of
            # reaction into the chassis per arm. Use --limp for gravity and
            # contact only.
            # [forward, yaw, boom, drum, shroud]. Boom at -1 is the stow angle,
            # not 0: zero is mid-range, which would drive the arm down into the
            # bed and plant the drums, which is the opposite of a static test.
            # The wheels are a velocity drive, so 0 commands them to hold still
            # rather than to freewheel.
            parked = torch.zeros(u.num_envs, 5, device=u.device)
            parked[:, 2] = -1.0
            policy, in_dim = (lambda _obs: parked), None
        elif args.dig:
            # boom_command_for_cut solves the arm angle that puts the SHROUD
            # `cut` metres under the surface and returns it already mapped to
            # the [-1, 1] action, so the depth asked for here is the depth the
            # reward is written against rather than a guess at a joint angle.
            cut = args.cut
            boom_cmd, angle = u.cfg.boom_command_for_cut(cut)
            if args.boom is not None:
                lo_a, hi_a = u.cfg.arm_range
                boom_cmd = max(-1.0, min(1.0, args.boom))
                angle = lo_a + 0.5 * (boom_cmd + 1.0) * (hi_a - lo_a)
                print(f"[play] --boom {args.boom:+.2f} overrides --cut: the "
                      f"geometry solve wanted {cut * 100:.1f} cm")
            ramp = max(1, int(args.dig_ramp / u.step_dt))
            # Lazily, like _wheel_radius: this script should not stop working
            # because a geometry constant moved.
            try:
                from luna_hifi_tasks.excavator.excavator_cfg import (  # noqa: PLC0415
                    MAX_DRUM_SPEED,
                    MAX_WHEEL_SPEED,
                )
            except Exception:  # noqa: BLE001
                MAX_DRUM_SPEED, MAX_WHEEL_SPEED = 2.9, 5.0
            # boom_command_for_cut places the SHROUD, outer radius 0.212,
            # while the thing that cuts is the rotor at tip radius 0.185. It
            # also assumes the wheels rest exactly on bed_top, but SPAWN_Z
            # carries 20 mm of clearance and -- now that the soil holds the
            # machine up instead of letting it sink -- it stays there. Both
            # errors lift the rotor, and together they are 47 mm: at the
            # task's own 4 cm cut the rotor tip ends up 7 mm ABOVE the bed and
            # the run excavates nothing, which reads exactly like soil that is
            # too hard to cut. Predict the tip here and say so before the run,
            # rather than after twenty minutes.
            try:
                from luna_hifi_tasks.excavator.excavator import (  # noqa: PLC0415
                    ARM_LEN,
                    CHASSIS_Z,
                    MAST_TOP_Z,
                    ROTOR_TIP_R,
                )

                chassis0 = float(u.cfg.bed_top) + CHASSIS_Z + 0.02
                tip = chassis0 + MAST_TOP_Z - ARM_LEN * math.sin(angle) - ROTOR_TIP_R
                gap = tip - float(u.cfg.bed_top)
                print(f"[play] DIG geometry: rotor tip lands at z={tip:.4f} "
                      f"against a bed surface of {u.cfg.bed_top:.4f}, "
                      f"{1000 * gap:+.0f} mm")
                if gap >= 0.0:
                    print("[play] *** the rotor is ABOVE the bed and will cut "
                          "NOTHING. Raise --cut or use --boom. ***")
            except Exception as exc:  # noqa: BLE001
                print(f"[play] could not predict the rotor tip: {exc!r}")
            print(
                f"[play] DIG: cut {cut * 100:.1f} cm -> boom command "
                f"{boom_cmd:+.3f} (arm {angle:+.4f} rad, "
                f"{math.degrees(angle):+.1f} deg), drums to "
                f"{MAX_DRUM_SPEED:.2f} rad/s, then forward {args.dig_speed:+.2f} "
                f"({args.dig_speed * MAX_WHEEL_SPEED * _wheel_radius():.2f} m/s) "
                f"after a {args.dig_ramp:.1f}s ramp"
            )
            # Lower and spin up BEFORE moving off. Driving into the bed with
            # the arm still falling plants the drums instead of cutting with
            # them, which is the failure this is meant to distinguish from a
            # machine that cannot cut.
            dig_a = torch.zeros(u.num_envs, 5, device=u.device)
            k = [0]

            def policy(_obs, _a=dig_a, _k=k, _b=boom_cmd, _r=ramp):
                f = min(1.0, _k[0] / _r)
                _k[0] += 1
                _a[:, 0] = args.dig_speed if f >= 1.0 else 0.0
                _a[:, 2] = -1.0 + f * (_b + 1.0)
                _a[:, 3] = f
                if _k[0] == _r + 1:
                    # The ramp is over and the machine is about to move off.
                    # Print where the arm and drums ACTUALLY got to, measured
                    # rather than derived, before 500 steps of output buries
                    # it. A drum still above the soil here means the run can
                    # say nothing about whether the soil can be cut.
                    print("[play] --- end of dig ramp, before moving off ---")
                    _attitude(u)
                    print(f"[play] undisturbed bed surface is z={u.cfg.bed_top:.4f}; "
                          f"a drum bottom above that never entered the soil")
                return _a

            in_dim = None
        else:
            policy, in_dim = _load_policy(str(u.device), ckpt)

        if args.limp:
            # Gains to zero on every joint, so the only forces on the machine
            # are gravity and contact. --park commands a setpoint on all four
            # controllers and they hold it; this removes them. The arms will
            # fall under their own weight, which is what an unpowered machine
            # does and is the point of the test.
            try:
                import torch as _t

                nj = u.robot.data.joint_pos.torch.shape[1]
                z = _t.zeros((u.num_envs, nj), device=u.device)
                u.robot.write_joint_stiffness_to_sim(z)
                u.robot.write_joint_damping_to_sim(z)
                print(f"[play] LIMP: stiffness and damping zeroed on {nj} joints. "
                      f"nothing is driven; arms will fall under gravity")
            except Exception as exc:  # noqa: BLE001
                print(f"[play] --limp failed, actuators are STILL driven: {exc!r}")

        dt = u.step_dt
        steps = int(args.seconds / dt)
        reset_steps: list[int] = []
        bodies: list = []
        import time as _time

        t_start = _time.perf_counter()
        t_mark = t_start
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
                    # Every body's world position, so the machine's ATTITUDE is
                    # recoverable afterwards. The chassis pose alone cannot show
                    # a pitch, and the heightmap is a plan view, so neither of
                    # the things already dumped could show the machine tipping --
                    # which is what it has been doing all along.
                    try:
                        bp = u.robot.data.body_pos_w.torch[0].detach().cpu().numpy()
                        bodies.append((bp - origin.detach().cpu().numpy()).copy())
                    except Exception:  # noqa: BLE001
                        pass
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

            if (i + 1) % 100 == 0:
                now = _time.perf_counter()
                sps = 100.0 / max(now - t_mark, 1e-9)
                t_mark = now
                print(f"[play] step {i + 1}/{steps}  {sps:6.2f} env-steps/s  "
                      f"eta {(steps - i - 1) / max(sps, 1e-9):6.0f}s", flush=True)

            if bool(dones.any()):
                reset_steps.append(i)
                print(f"[play] t={i*dt:5.2f}s episode ended "
                      f"(shape_done={int(scalar('_shape_done'))}, "
                      f"escaped={int(scalar('_escaped'))})")

        if args.probe:
            _attitude(u)
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
                body_pos=(np.stack(bodies).astype("float32") if bodies
                          else np.zeros((0, 0, 3), dtype="float32")),
                # Plain unicode, NOT dtype=object. An object array is pickled
                # into the npz, and a pickle written by the venv's numpy 2.x
                # cannot be read by the system numpy 1.x -- it fails with
                # "No module named 'numpy._core'", which makes the dump
                # unreadable by exactly the interpreter the renderers run under.
                body_names=np.asarray(
                    list(getattr(u.robot.data, "body_names", []) or []), dtype="U64"
                ),
                wheel_radius=_wheel_radius(),
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

        wall = _time.perf_counter() - t_start
        sps = steps / max(wall, 1e-9)
        print(f"[play] {steps} steps in {wall:.1f}s = {sps:.2f} env-steps/s "
              f"at {u.num_envs} env(s), {sps * dt:.2f}x realtime")
        # What that means for training, which is the number that decides whether
        # the current solver settings are affordable at all.
        for n_env in (16, 64):
            # One grid is shared across envs, so per-env-step cost falls with
            # env count; this scales only the per-env work and is therefore
            # pessimistic, which is the right direction for a budget.
            eps_steps = int(26.0 / dt)
            iters = 400
            total = eps_steps * iters * n_env
            hrs = total / max(sps * n_env, 1e-9) / 3600.0
            print(f"[play]   at {n_env:3d} envs: {iters} iterations x {eps_steps} "
                  f"steps = {total / 1e6:.1f}M env-steps, ~{hrs:.1f} h "
                  f"(pessimistic: assumes no per-env speedup)")

        last = rows[-1]
        peak = max(r["payload_kg"] for r in rows)

        # Sinkage, printed here rather than left in the CSV. This is the
        # headline number of a parked run and leaving it out of the summary
        # made several runs unreadable without a second pass over the trace.
        z = [r["chassis_z"] for r in rows]
        tail = z[-min(50, len(z)):]
        rate = (tail[0] - tail[-1]) / (max(len(tail) - 1, 1) * dt)
        print(f"[play] chassis_z {z[0]:.4f} -> {z[-1]:.4f} m, min {min(z):.4f}: "
              f"sank {1000 * (z[0] - min(z)):.0f} mm, "
              f"last {len(tail)} steps {1000 * (tail[0] - tail[-1]):+.1f} mm "
              f"({1000 * rate:+.2f} mm/s, ~0 means it has settled)")

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
