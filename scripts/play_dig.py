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
    p.add_argument("--trace", default="dig_trace.csv", help="per-step CSV")
    p.add_argument("--frames", default=None,
                   help="directory for PNGs, if the env can render to an array. "
                        "The Newton backend hands back nothing, so prefer --dump")
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


def main(argv=None) -> int:
    args = _parse(argv)
    torch.manual_seed(0)

    ckpt = _resolve_checkpoint(args.run, args.checkpoint)
    print(f"[play] checkpoint {ckpt}")

    from luna_hifi_tasks.excavator.excavator_cfg import usd_status
    problem = usd_status()
    if problem is not None:
        print(f"\n[asset] {problem}\n")
        return 2

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs

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
        policy, in_dim = _load_policy(str(u.device), ckpt)

        dt = u.step_dt
        steps = int(args.seconds / dt)
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
        if actor_obs.shape[-1] != in_dim:
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
                print(f"[play] t={i*dt:5.2f}s episode ended "
                      f"(shape_done={int(scalar('_shape_done'))}, "
                      f"escaped={int(scalar('_escaped'))})")

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
