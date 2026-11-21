# scripts/site_run.py
#
# Dig a trench with the trained policies on the RFT soil (Luna-Excavator-Site):
# the planner (luna_hifi_tasks/planner/site.py) picks each pass, the navigate
# policy drives to it, the excavate policy digs it, and the drum is emptied on
# a pile past the trench. Writes a line per trim to the terminal and to --log,
# and heightmap frames to --dump for render_dig.py.
#
#   bash workstation/train.sh -p scripts/site_run.py --video site_video
#   python3 scripts/render_dig.py site.npz --stride 4
#
# The dig checkpoint defaults to the newest intact one of Luna-Excavator-Excavate-RFT,
# the navigate one to Luna-Excavator-Navigate-RFT's (fine-tuned on the soil), else
# Luna-Excavator-Navigate's.

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)
from luna_hifi_tasks.excavator.excavator import WHEEL_RADIUS
from luna_hifi_tasks.excavator.site.site_env import run_site
from luna_hifi_tasks.planner.site import SiteCfg, trench
from play_dig import _load_policy, _resolve_checkpoint

TASK = "Luna-Excavator-Site"
# The recorder holds a clip's frames in memory: a clip a minute, every third step,
# played back at the step rate, so three times real time; joined at the end.
CLIP_STEPS, FRAME_STRIDE = 1500, 3


def _parse():
    p = argparse.ArgumentParser(description="Dig a trench with the trained excavate and navigate policies.")
    p.add_argument("--length", type=float, default=2.4, help="trench length [m]")
    p.add_argument("--depth", type=float, default=0.15, help="trench depth [m]")
    p.add_argument("--dig_checkpoint", default=None, help="excavate model file; default the newest intact RFT one")
    p.add_argument("--nav_checkpoint", default=None,
                   help="navigate model file; default the newest intact one fine-tuned on the soil, else navigate's")
    p.add_argument("--minutes", type=float, default=30.0, help="simulated time limit")
    p.add_argument("--video", default=None, help="record an mp4 into this directory, from the Newton viewer")
    p.add_argument("--eye", type=float, nargs=3, default=None, help="video camera position; default over the site")
    p.add_argument("--lookat", type=float, nargs=3, default=None, help="video camera target")
    p.add_argument("--dump", default="site.npz", help="heightmap frames for render_dig.py")
    p.add_argument("--dump_every", type=int, default=10, help="steps between saved frames")
    p.add_argument("--log", default="site_log.csv", help="one row per trim")
    p.add_argument("--trace", default="site_trace.csv", help="one row per step: the drive command and what the machine did")
    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=["newton_gl"])
    # As play_dig: this script's flags off the command line, the rest left for Hydra.
    args, hydra_args = setup_preset_cli(p)
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def _newest_navigate() -> Path:
    """The newest intact navigate checkpoint: fine-tuned on the soil if there is one."""
    try:
        return _resolve_checkpoint("Luna-Excavator-Navigate-RFT", None, None)
    except FileNotFoundError:
        return _resolve_checkpoint("Luna-Excavator-Navigate", None, None)


def main() -> int:
    args = _parse()
    torch.manual_seed(0)
    dig_ckpt = Path(args.dig_checkpoint) if args.dig_checkpoint else _resolve_checkpoint("Luna-Excavator-Excavate-RFT", None, None)
    nav_ckpt = Path(args.nav_checkpoint) if args.nav_checkpoint else _newest_navigate()
    print(f"[site] dig policy {dig_ckpt}\n[site] navigate policy {nav_ckpt}")

    env_cfg, _ = resolve_task_config(TASK, "")
    env_cfg.trench_length = args.length
    env_cfg.__post_init__()
    if args.video:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        x = 0.5 * (env_cfg.bed_x[0] + env_cfg.bed_x[1])
        env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(
            eye=tuple(args.eye) if args.eye else (x, -6.5, 7.0),
            lookat=tuple(args.lookat) if args.lookat else (x, 1.0, 0.0),
        )]
        env_cfg.soil_marker = True
        env_cfg.video_recorders = [VideoRecorderCfg(
            source="visualizer:newton", output_dir=args.video, video_length=CLIP_STEPS, video_interval=CLIP_STEPS,
            frame_stride=FRAME_STRIDE, fps=round(1.0 / (env_cfg.sim.dt * env_cfg.decimation)),
        )]
    earlier = set(Path(args.video).glob("clip_*.mp4")) if args.video else set()
    args.device = env_cfg.sim.device
    env_cfg.validate()

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        dig = _load_policy(str(u.device), dig_ckpt)
        nav = _load_policy(str(u.device), nav_ckpt)
        site = trench(SiteCfg(), args.length, args.depth)
        print(f"[site] trench {args.length:g} m x {u.cfg.work_area_width:g} m x {args.depth:g} m: "
              f"{len(site.height[0])} patches, {-site.target[0][0]} passes of {100 * site.cfg.voxel:g} cm each")

        trench_cells = u.trench_mask()
        fresh = u._fresh_bed
        kg_per_m = u.cfg.voxel_size ** 2 * u.cfg.soil_density
        beds, poses, bodies, trace, steps = [], [], [], [], [0]

        def record(u) -> None:
            steps[0] += 1
            d = u.robot.data
            o = u.scene.env_origins[0]
            root = d.root_pos_w.torch[0] - o
            wheels = d.joint_vel.torch[0] * WHEEL_RADIUS
            trace.append({
                "t": round(steps[0] * u.step_dt, 2), "mode": u.mode,
                "x": round(float(root[0]), 3), "y": round(float(root[1]), 3),
                "yaw_deg": round(math.degrees(float(u._base_yaw()[0])), 1),
                "goal_m": round(u.goal_error()[0], 3) if u.mode == "drive" else "",
                "cmd_mps": round(float(u._forward_cmd[0]), 3), "cmd_yaw_rate": round(float(u._yaw_cmd[0]), 3),
                "speed_mps": round(float(d.root_lin_vel_b.torch[0, 0]), 3),
                "side_mps": round(float(d.root_lin_vel_b.torch[0, 1]), 3),
                "yaw_rate": round(float(d.root_ang_vel_b.torch[0, 2]), 3),
                "rim_left_mps": round(float(wheels[u._left_ids].mean()), 3),
                "rim_right_mps": round(float(wheels[u._right_ids].mean()), 3),
            })
            if steps[0] % args.dump_every:
                return
            bed = u._bed_grid[0]
            removed = float(((fresh - bed).clamp(min=0.0) * trench_cells).sum()) * kg_per_m
            beds.append(bed.cpu().numpy().copy())
            bodies.append((u.robot.data.body_pos_w.torch[0] - o).cpu().numpy().copy())
            poses.append([float(root[0]), float(root[1]), float(root[2]), float(u._base_yaw()[0]),
                          steps[0] * u.step_dt, removed, float(u._fill_kg[0, 0])])

        with torch.inference_mode():
            rows, ending = run_site(env, dig, nav, site, record, args.minutes)
        print(f"[site] {ending}: {site.summary()}")
        print(f"[site] patch heights, cm: {[100 * h * site.cfg.voxel for h in site.height[0]]}")
        names = np.asarray(u.robot.body_names, dtype="U64")
        env.close()

    if args.video:
        from moviepy.editor import VideoFileClip, concatenate_videoclips

        clips = sorted(set(Path(args.video).glob("clip_*.mp4")) - earlier)
        if clips:
            out = f"{args.video.rstrip('/')}.mp4"
            concatenate_videoclips([VideoFileClip(str(c)) for c in clips]).write_videofile(
                out, codec="libx264", audio=False, logger=None)
            print(f"[site] video {out}: {len(clips)} clips, {FRAME_STRIDE}x real time")

    if beds:
        np.savez_compressed(
            args.dump,
            bed=np.stack(beds).astype("float32"),
            pose=np.asarray(poses, dtype="float32"),
            work_mask=trench_cells.cpu().numpy(),
            body_pos=np.stack(bodies).astype("float32"),
            body_names=names,
            wheel_radius=WHEEL_RADIUS,
            voxel=float(u.cfg.voxel_size),
            lower=np.asarray(u.cfg.bed_grid_lower, dtype="float32"),
            bed_x=np.asarray(u.cfg.bed_x, dtype="float32"),
            bed_y=np.asarray(u.cfg.bed_y, dtype="float32"),
            drum_capacity_kg=float(u.cfg.drum_capacity_kg),
        )
        print(f"[site] wrote {len(beds)} heightmap frames to {args.dump}")
    if trace:
        with open(args.trace, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(trace[0]))
            w.writeheader()
            w.writerows(trace)
        print(f"[site] wrote {len(trace)} steps to {args.trace}")
    if rows:
        flat = [{f"{k}_{kk}" if isinstance(v, dict) else k: vv
                 for k, v in r.items() for kk, vv in (v.items() if isinstance(v, dict) else [(None, v)])} for r in rows]
        keys = list(dict.fromkeys(k for r in flat for k in r))
        with open(args.log, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(flat)
        print(f"[site] wrote {len(rows)} trims to {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
