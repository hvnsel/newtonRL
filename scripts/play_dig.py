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
                   help="directory for PNGs, if the env can render to an array")
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

    The state dict already holds everything needed. Layer widths come from the
    weight shapes, so this does not care what actor_hidden_dims says, and the
    observation normaliser is read from the same file as the weights it was
    fitted alongside. ELU is the activation the config asks for; there is no
    way to recover it from shapes, so it is the one assumption here, and the
    layer list is printed so a mismatch is visible rather than silent.
    """
    import torch.nn as nn

    blob = torch.load(str(ckpt), map_location=device, weights_only=False)
    state = blob.get("model_state_dict", blob)

    # actor.0.weight, actor.2.weight, ... -- the even indices are the Linears.
    widths: dict[int, tuple[int, int]] = {}
    for key, value in state.items():
        parts = key.split(".")
        if parts[0] == "actor" and parts[-1] == "weight" and value.dim() == 2:
            widths[int(parts[1])] = tuple(value.shape)
    if not widths:
        raise SystemExit(
            f"no actor.*.weight in {ckpt}; it holds {sorted(state)[:12]}"
        )

    layers: list[nn.Module] = []
    order = sorted(widths)
    for n, idx in enumerate(order):
        out_f, in_f = widths[idx]
        layers.append(nn.Linear(in_f, out_f))
        if n != len(order) - 1:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)

    # Sequential numbers its own children 0,1,2..., which matches the saved
    # indices only when every gap is one activation. Map by position instead.
    own = [i for i, m in enumerate(actor) if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for mine, theirs in zip(own, order):
            actor[mine].weight.copy_(state[f"actor.{theirs}.weight"])
            actor[mine].bias.copy_(state[f"actor.{theirs}.bias"])
    actor.to(device).eval()

    # The normaliser the actor was trained behind, if the run used one.
    # EmpiricalNormalization has been spelt both mean/var and _mean/_var, and
    # stores them shaped (1, D), so match on the trimmed leaf and flatten.
    mean = var = None
    for key, value in state.items():
        if "normalizer" not in key or "critic" in key:
            continue
        leaf = key.rsplit(".", 1)[-1].lstrip("_")
        if leaf == "mean":
            mean = value.to(device).flatten()
        elif leaf == "var":
            var = value.to(device).flatten()
    norm = "none"
    if mean is not None and var is not None:
        norm = f"mean/var over {mean.numel()} dims"

    in_dim = widths[order[0]][1]
    print(f"[play] actor {in_dim} -> "
          f"{' -> '.join(str(widths[i][0]) for i in order)}, elu, obs norm: {norm}")
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
            below = float(u._below_prev[0]) * soil
            rows.append({
                "t": round(i * dt, 4),
                "removal_kg": round(float(u._removed_kg[0]), 3),
                "payload_kg": round(float(u._payload_kg[0]), 3),
                "cut_total_kg": round(float(u._cut_total_kg[0]), 3),
                "below_kg": round(below, 3),
                "advance_mps": round(float(u.robot.data.root_lin_vel_b.torch[0, 0]), 4),
                "chassis_z": round(float(root[0, 2] - u.scene.env_origins[0, 2]), 4),
                "particle_max_m": round(float(u._particle_max_m[0]), 2),
                "shape_done": int(u._shape_done[0]),
                "escaped": int(u._escaped[0]),
            })

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
                      f"(shape_done={int(u._shape_done[0])}, escaped={int(u._escaped[0])})")

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
        print(f"[play] end: removal {last['removal_kg']} kg, payload {last['payload_kg']} kg, "
              f"hole {last['below_kg']} kg, particle_max {last['particle_max_m']} m")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
