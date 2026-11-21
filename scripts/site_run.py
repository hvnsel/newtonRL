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
# The dig and navigate checkpoints default to the newest intact one of
# Luna-Excavator-Excavate-RFT and Luna-Excavator-Navigate.

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_task_config

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)
from luna_hifi_tasks.excavator.excavator import WHEEL_RADIUS
from luna_hifi_tasks.excavator.site.site_env import run_site
from luna_hifi_tasks.planner.site import SiteCfg, trench
from play_dig import _load_policy, _resolve_checkpoint

TASK = "Luna-Excavator-Site"


def _parse():
    p = argparse.ArgumentParser(description="Dig a trench with the trained excavate and navigate policies.")
    p.add_argument("--length", type=float, default=2.4, help="trench length [m]")
    p.add_argument("--depth", type=float, default=0.15, help="trench depth [m]")
    p.add_argument("--dig_checkpoint", default=None, help="excavate model file; default the newest intact RFT one")
    p.add_argument("--nav_checkpoint", default=None, help="navigate model file; default the newest intact one")
    p.add_argument("--minutes", type=float, default=30.0, help="simulated time limit")
    p.add_argument("--video", default=None, help="record an mp4 into this directory, from the Newton viewer")
    p.add_argument("--eye", type=float, nargs=3, default=None, help="video camera position; default over the site")
    p.add_argument("--lookat", type=float, nargs=3, default=None, help="video camera target")
    p.add_argument("--dump", default="site.npz", help="heightmap frames for render_dig.py")
    p.add_argument("--dump_every", type=int, default=10, help="steps between saved frames")
    p.add_argument("--log", default="site_log.csv", help="one row per trim")
    add_launcher_args(p)
    return p.parse_args()


def main() -> int:
    args = _parse()
    torch.manual_seed(0)
    dig_ckpt = Path(args.dig_checkpoint) if args.dig_checkpoint else _resolve_checkpoint("Luna-Excavator-Excavate-RFT", None, None)
    nav_ckpt = Path(args.nav_checkpoint) if args.nav_checkpoint else _resolve_checkpoint("Luna-Excavator-Navigate", None, None)
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
            source="visualizer:newton", output_dir=args.video, video_interval=0,
            video_length=int(args.minutes * 60.0 / (env_cfg.sim.dt * env_cfg.decimation)),
        )]
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
        beds, poses, bodies, steps = [], [], [], [0]

        def record(u) -> None:
            steps[0] += 1
            if steps[0] % args.dump_every:
                return
            o = u.scene.env_origins[0]
            root = u.robot.data.root_pos_w.torch[0] - o
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
