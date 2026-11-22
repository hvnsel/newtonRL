# scripts/site_run.py
#
# Dig a trench, with --pit a pit entered down a staircase ramp, or with --l an
# L or --hole a square hole cut in strokes along x and y, with the trained
# policies on the RFT soil (Luna-Excavator-Site): the planner (luna_hifi_tasks/
# planner/site.py, or strokes.py for strokes) picks each pass or stroke and its
# level, the navigate policy drives to it, the excavate policy digs it, and the
# drum is emptied on a pile beside the site. Writes a line per trim or stroke
# to the terminal, and with --video the run as an mp4, the viewer rendering only
# the frames the video keeps.
#
#   bash workstation/train.sh -p scripts/site_run.py --video site_video
#   bash workstation/train.sh -p scripts/site_run.py --pit --step 0.075 --video pit_video
#   bash workstation/train.sh -p scripts/site_run.py --l --video l_video
#   bash workstation/train.sh -p scripts/site_run.py --hole --video hole_video
#
# The dig checkpoint defaults to the newest intact one of Luna-Excavator-Excavate-RFT,
# the navigate one to Luna-Excavator-Navigate-RFT's (fine-tuned on the soil), else
# Luna-Excavator-Navigate's.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)
from luna_hifi_tasks.excavator.site.site_env import RING, box_corner, run_site, run_strokes
from luna_hifi_tasks.planner.site import SiteCfg, pit, trench
from luna_hifi_tasks.planner.strokes import Heights, StrokeCfg, StrokeSite, hole_rects, hole_strokes, l_rects, l_strokes
from play_dig import _load_policy, _resolve_checkpoint

TASK = "Luna-Excavator-Site"
# The recorder holds a clip's frames in memory: a clip a minute, every
# --frame_stride-th step, played back at the step rate; joined at the end.
CLIP_STEPS = 1500


def _parse():
    p = argparse.ArgumentParser(description="Dig a trench with the trained excavate and navigate policies.")
    p.add_argument("--pit", action="store_true", help="a pit entered down a staircase ramp, not a trench")
    p.add_argument("--l", action="store_true", help="an L cut in strokes along x and y, not a trench")
    p.add_argument("--hole", action="store_true", help="a square hole cut in strokes along x and y, not a trench")
    p.add_argument("--size", type=float, default=None, help="L or hole: its side [m]; default an L 3.325, a hole 1.9")
    p.add_argument("--thickness", type=float, default=1.425, help="L: its arms across [m]; 1.5 drum widths")
    p.add_argument("--length", type=float, default=None, help="length [m]; default 2.4, a pit 4.0")
    p.add_argument("--depth", type=float, default=None, help="depth [m]; default 0.15, a pit 0.30")
    p.add_argument("--step", type=float, default=0.075, help="pit: how much deeper each patch down the ramp [m]")
    p.add_argument("--lanes", type=int, default=5, help="pit: lanes across, the side ones shelves")
    p.add_argument("--dig_checkpoint", default=None, help="excavate model file; default the newest intact RFT one")
    p.add_argument("--nav_checkpoint", default=None,
                   help="navigate model file; default the newest intact one fine-tuned on the soil, else navigate's")
    p.add_argument("--minutes", type=float, default=None, help="simulated time limit; default 30, a pit 300, an L 120, a hole 60")
    p.add_argument("--video", default=None, help="record an mp4 into this directory, from the Newton viewer")
    p.add_argument("--eye", type=float, nargs=3, default=None, help="video camera position; default over the site")
    p.add_argument("--lookat", type=float, nargs=3, default=None, help="video camera target")
    p.add_argument("--frame_stride", type=int, default=None,
                   help="video: every n-th step, so n times real time; default 3, a pit and an L 9, a hole 6")
    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=["newton_gl"])
    # As play_dig: this script's flags off the command line, the rest left for Hydra.
    args, hydra_args = setup_preset_cli(p)
    sys.argv = [sys.argv[0]] + hydra_args
    # Defaults for a trench, a pit, an L and a hole.
    mode = 2 if args.l else 3 if args.hole else 1 if args.pit else 0
    for key, values in (("length", (2.4, 4.0, 0.0, 0.0)), ("depth", (0.15, 0.30, 0.15, 0.15)),
                        ("minutes", (30.0, 300.0, 120.0, 60.0)), ("frame_stride", (3, 9, 9, 6)),
                        ("size", (0.0, 0.0, 3.325, 1.9))):
        if getattr(args, key) is None:
            setattr(args, key, values[mode])
    args.strokes = args.l or args.hole
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
    env_cfg.pit_length = args.length
    env_cfg.pit_lanes = args.lanes if args.pit else 1
    env_cfg.box_size = args.size if args.strokes else 0.0
    env_cfg.__post_init__()
    if args.video:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        # From the -y side, over the middle of the bed, far enough out for all
        # of it; for strokes, over the site and the ring road round it.
        (x0, x1), (y0, y1) = env_cfg.bed_x, env_cfg.bed_y
        x, y, span = 0.5 * (x0 + x1), 0.5 * (y0 + y1), max(x1 - x0, y1 - y0)
        eye = (x, y0 - 0.45 * span, 0.75 * span)
        if args.strokes:
            bx, by = box_corner(env_cfg)
            x, y, span = bx + 0.5 * args.size, by + 0.5 * args.size, args.size + 2.0 * RING
            eye = (x, y - 0.95 * span, 0.75 * span)
        env_cfg.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(
            eye=tuple(args.eye) if args.eye else eye,
            lookat=tuple(args.lookat) if args.lookat else (x, y, 0.0),
        )]
        env_cfg.soil_marker = True
        # The viewer renders, and the soil is drawn, only for the frames the video keeps.
        env_cfg.sim.render_interval = env_cfg.decimation * args.frame_stride
        env_cfg.video_recorders = [VideoRecorderCfg(
            source="visualizer:newton", output_dir=args.video, video_length=CLIP_STEPS, video_interval=CLIP_STEPS,
            frame_stride=args.frame_stride, fps=round(1.0 / (env_cfg.sim.dt * env_cfg.decimation)),
        )]
    earlier = set(Path(args.video).glob("clip_*.mp4")) if args.video else set()
    args.device = env_cfg.sim.device
    env_cfg.validate()

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg)
        u = env.unwrapped
        dig = _load_policy(str(u.device), dig_ckpt)
        nav = _load_policy(str(u.device), nav_ckpt)
        if args.strokes:
            corner = box_corner(u.cfg)
            c = u.cfg
            cfg = StrokeCfg(width=c.work_area_width, ahead=c.work_area_length, behind=c.work_area_behind,
                            dig_reach=u.dig_reach)
            cfg.run_in = RING + cfg.wheelbase       # lined up on the ring road, its rear wheels
            if args.l:
                strokes, area = l_strokes(cfg, corner, args.size, args.thickness), l_rects(corner, args.size, args.thickness)
                what = f"L {args.size:g} m a side, arms {args.thickness:g} m across"
            else:
                strokes, area = hole_strokes(cfg, corner, args.size), hole_rects(corner, args.size)
                what = f"hole {args.size:g} m square"
            site = StrokeSite(cfg, strokes, -args.depth, Heights(*u.heights()))
            print(f"[site] {what}, {args.depth:g} m deep: {len(site.strokes)} strokes along x and y, "
                  f"{100 * cfg.layer:g} cm a layer")
        elif args.pit:
            site = pit(SiteCfg(), args.length, args.depth, args.step, args.lanes)
            print(f"[site] pit {args.length:g} m x {args.lanes * u.cfg.work_area_width:g} m x {args.depth:g} m, "
                  f"{args.lanes} lanes of {len(site.height[0])} patches, the side lanes shelves; "
                  f"down a ramp {100 * args.step:g} cm a patch")
        else:
            site = trench(SiteCfg(), args.length, args.depth)
            print(f"[site] trench {args.length:g} m x {u.cfg.work_area_width:g} m x {args.depth:g} m: "
                  f"{len(site.height[0])} patches, {-site.target[0][0]} passes of {100 * site.cfg.voxel:g} cm each")

        if not args.strokes:
            area = [(u.pit_x, u.pit_x + args.length, -u.half_width, u.half_width)]
        # The ground is shaded by depth over the cut and its walls; elsewhere, ruts and all, the soil's colour.
        u.shade_area = u.rects_mask([(x0 - 0.3, x1 + 0.3, y0 - 0.3, y1 + 0.3) for x0, x1, y0, y1 in area])

        with torch.inference_mode():
            if args.strokes:
                _, ending = run_strokes(env, dig, nav, site, max_minutes=args.minutes)
            else:
                _, ending = run_site(env, dig, nav, site, max_minutes=args.minutes)
        if args.strokes:
            print(f"[site] {ending}")
            for i, s in enumerate(site.strokes):
                print(f"[site] stroke {s.name}: {100 * site.level(i):+.1f} cm")
        else:
            print(f"[site] {ending}: {site.summary()}")
            for iy, lane in enumerate(site.height):
                print(f"[site] lane {iy} heights, cm: {[round(100 * h * site.cfg.voxel, 1) for h in lane]}")
        env.close()

    if args.video:
        from moviepy.editor import VideoFileClip, concatenate_videoclips

        clips = sorted(set(Path(args.video).glob("clip_*.mp4")) - earlier)
        if clips:
            out = f"{args.video.rstrip('/')}.mp4"
            concatenate_videoclips([VideoFileClip(str(c)) for c in clips]).write_videofile(
                out, codec="libx264", audio=False, logger=None)
            print(f"[site] video {out}: {len(clips)} clips, {args.frame_stride}x real time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
