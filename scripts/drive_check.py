"""Does the machine drive?

Commands a constant action and reports how far the chassis actually travels,
in the env frame, and how far it turns. Compares both against what the
commanded speed and yaw rate should produce over the same number of steps.

    isaaclab -p scripts/drive_check.py --task Luna-Excavator-Navigate \
        --num_envs 16 --steps 749 --forward 1.0

Turning on the spot, with the wheel motors' torque cap overridden:

    isaaclab -p scripts/drive_check.py --task Luna-Excavator-Navigate \
        --num_envs 16 --steps 125 --forward 0 --yaw 1 --wheel_effort 240

A policy-free measurement: whatever the reward is doing, this is the ceiling
the machine can reach.
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Luna-Excavator-Navigate")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=749)
    p.add_argument("--forward", type=float, default=1.0, help="action[0] in [-1, 1]")
    p.add_argument("--yaw", type=float, default=0.0, help="action[1] in [-1, 1]")
    p.add_argument("--wheel_effort", type=float, default=None,
                   help="override the wheel motors' torque cap, N.m per wheel")
    p.add_argument("--terrain_rows", type=int, default=None)
    p.add_argument("--terrain_cols", type=int, default=None)
    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=["newton_gl"])
    # resolve_task_config runs Hydra, which re-reads sys.argv, so this
    # script's own flags are split off first and the remainder handed over.
    args, hydra_args = setup_preset_cli(p, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def main(argv=None) -> int:
    args = _parse(argv)
    torch.manual_seed(0)

    from luna_hifi_tasks.excavator.excavator import WHEEL_RADIUS
    from luna_hifi_tasks.excavator.excavator_cfg import MAX_WHEEL_SPEED, MAX_YAW_SPEED, usd_status
    from luna_hifi_tasks.excavator.mdp.terrain import yaw_from_quat

    problem = usd_status()
    if problem is not None:
        print(f"\n[asset] {problem}\n")
        return 2

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs
    if args.wheel_effort is not None:
        wheels = env_cfg.scene.excavator.actuators["wheels"]
        print(f"[drive_check] wheel torque cap {wheels.effort_limit_sim} -> {args.wheel_effort} N.m")
        wheels.effort_limit_sim = args.wheel_effort

    gen = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
    if gen is not None:
        if args.terrain_rows is not None:
            gen.num_rows = args.terrain_rows
        if args.terrain_cols is not None:
            gen.num_cols = args.terrain_cols
        env_cfg.scene.terrain.max_init_terrain_level = min(
            env_cfg.scene.terrain.max_init_terrain_level, gen.num_rows - 1
        )

    args.device = env_cfg.sim.device
    env_cfg.validate()

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        u = env.unwrapped
        dev = u.device
        dt = u.step_dt
        # What the simulation actually holds, which is what --wheel_effort has
        # to have changed for the comparison to mean anything.
        caps = u.robot.data.joint_effort_limits.torch[0, u._wheel_ids]
        print(f"[drive_check] wheel torque cap in the sim: {[round(float(c), 1) for c in caps]} N.m")

        n_act = env.action_space.shape[-1]
        action = torch.zeros(u.num_envs, n_act, device=dev)
        action[:, 0] = args.forward
        if n_act > 1:
            action[:, 1] = args.yaw

        env.reset()

        zero = torch.zeros_like(action)
        for _ in range(25):
            env.step(zero)

        origins = u.scene.env_origins
        start = (u.robot.data.root_pos_w.torch - origins).clone()
        # Heading change summed step by step, so a full turn and more reads
        # as such instead of wrapping back to zero.
        prev_yaw = yaw_from_quat(u.robot.data.root_quat_w.torch)
        turned = torch.zeros(u.num_envs, device=dev)

        peak_speed = torch.zeros(u.num_envs, device=dev)
        for i in range(args.steps):
            env.step(action)
            v = u.robot.data.root_lin_vel_b.torch[:, 0]
            peak_speed = torch.maximum(peak_speed, v)
            yaw = yaw_from_quat(u.robot.data.root_quat_w.torch)
            step_turn = yaw - prev_yaw
            turned += torch.atan2(torch.sin(step_turn), torch.cos(step_turn))
            prev_yaw = yaw
            if i % 100 == 0:
                pos = u.robot.data.root_pos_w.torch - origins
                moved = torch.linalg.norm((pos - start)[:, :2], dim=-1)
                print(
                    f"  step {i:4d}  moved {moved.mean():6.2f} m  "
                    f"speed {v.mean():5.2f} m/s  turned {torch.rad2deg(turned).mean():7.1f} deg"
                )

        end = (u.robot.data.root_pos_w.torch - origins).clone()
        moved = torch.linalg.norm((end - start)[:, :2], dim=-1)

        commanded = args.forward * MAX_WHEEL_SPEED * WHEEL_RADIUS
        ideal = abs(commanded) * dt * args.steps
        yaw_cmd = args.yaw * MAX_YAW_SPEED
        turn_ideal = abs(yaw_cmd) * dt * args.steps
        turn = turned.abs()

        print("\n=== drive check ===")
        print(f"  action            forward={args.forward}  yaw={args.yaw}")
        print(f"  commanded speed   {commanded:6.2f} m/s")
        print(f"  policy step dt    {dt:6.4f} s  x {args.steps} steps")
        print(f"  distance if ideal {ideal:6.2f} m")
        print(f"  distance measured {moved.mean():6.2f} m  "
              f"(min {moved.min():.2f}, max {moved.max():.2f})")
        print(f"  peak forward speed{peak_speed.mean():6.2f} m/s")
        print(f"  fraction of ideal {float(moved.mean() / max(ideal, 1e-6)):6.1%}")
        print(f"  commanded yaw rate{yaw_cmd:6.2f} rad/s")
        print(f"  turn if ideal     {torch.rad2deg(torch.tensor(turn_ideal)):7.1f} deg")
        print(f"  turn measured     {torch.rad2deg(turn).mean():7.1f} deg  "
              f"(min {torch.rad2deg(turn).min():.1f}, max {torch.rad2deg(turn).max():.1f}), "
              f"{float(turn.mean() / (dt * args.steps)):.2f} rad/s")
        if turn_ideal > 0:
            print(f"  turn fraction     {float(turn.mean() / turn_ideal):6.1%}")

        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
