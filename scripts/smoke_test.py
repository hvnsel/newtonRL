# scripts/smoke_test.py
#
# Run this before the first training job, on a machine with Isaac Lab:
#
#   isaaclab -p scripts/smoke_test.py --task Luna-Excavator-Navigate --num_envs 8
#   isaaclab -p scripts/smoke_test.py --task Luna-Excavator-Excavate --num_envs 2
#
# It builds the env exactly the way `isaaclab zero_agent` does, steps it a few
# hundred times, and asserts the things that fail SILENTLY in training:
#
#   * observation widths match their declared specs (a mismatch here would
#     raise in assembly anyway; this just makes it the first thing you see)
#   * every observation is finite
#   * the terrain scan is not a constant (a constant scan means the ray caster
#     is not hitting the ground, or the particle rasteriser sees no particles)
#   * on the MPM tier, drum fill RISES when the arms are lowered and the drums
#     spin over the bed. If it stays at zero, the particle adapter is reading
#     the wrong thing, and a training run would look exactly like a policy
#     that never learns to dig
#   * the machine is upright after settling (projected gravity z near -1). If
#     it reads +1 the spawn quaternion is in the wrong order (see the note at
#     the bottom of excavator_cfg.py)
#
# It also prints the articulation's body and joint names and every prim path
# that matched the coupler / ray-caster regexes, so the assumptions about how
# the MJCF converter nests bodies can be checked in one look.

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_task_config

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)


def _parse(argv):
    p = argparse.ArgumentParser(description="Excavator environment smoke test.")
    p.add_argument("--task", required=True, help="Luna-Excavator-Navigate or Luna-Excavator-Excavate")
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--steps", type=int, default=300)
    add_launcher_args(p)
    return p.parse_args(argv)


def _check(cond: bool, msg: str, failures: list[str]) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def main(argv=None) -> int:
    args = _parse(argv)
    torch.manual_seed(0)

    env_cfg, _ = resolve_task_config(args.task, "")
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    args.device = env_cfg.sim.device
    env_cfg.validate()

    failures: list[str] = []
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        u = env.unwrapped
        is_dig = "Excavate" in args.task

        print("\n=== articulation ===")
        print("  bodies:", u.robot.body_names)
        print("  joints:", u.robot.joint_names)
        print("  wheel ids", u._wheel_ids, "arm ids", u._arm_ids, "drum ids", u._drum_ids,
              "drum body ids", u._drum_body_ids)
        try:
            import isaacsim.core.utils.prims as prim_utils  # type: ignore

            paths = [p.GetPath().pathString for p in prim_utils.get_all_matching_child_prims("/World/envs/env_0")]
            hits = [p for p in paths if "wheel" in p or "drum" in p or "chassis" in p]
            print("  env_0 prims with wheel/drum/chassis in the path:")
            for h in hits[:24]:
                print("   ", h)
        except Exception as exc:  # kit-less path: no prim utils, that is fine
            print(f"  (prim listing unavailable here: {type(exc).__name__})")

        obs, _ = env.reset()
        pol, crit = obs["policy"], obs["critic"]
        print("\n=== spaces ===")
        print("  policy", tuple(pol.shape), " critic", tuple(crit.shape), " action", env.action_space.shape)
        spec = u.cfg.observation_space
        _check(pol.shape[1] == spec, f"policy obs width {pol.shape[1]} == cfg.observation_space {spec}", failures)
        _check(crit.shape[1] == u.cfg.state_space, f"critic width {crit.shape[1]} == cfg.state_space {u.cfg.state_space}", failures)

        # settle with zero actions, then read attitude
        zero = torch.zeros(u.num_envs, env.action_space.shape[-1], device=u.device)
        for _ in range(50):
            obs, *_ = env.step(zero)
        g = u.robot.data.projected_gravity_b.torch[:, 2]
        print(f"\n=== attitude after 50 zero-action steps ===\n  projected_gravity_b z: {g.tolist()[:4]}")
        _check(bool((g < -0.9).all()), "upright: projected_gravity_b z < -0.9 for all envs (spawn quaternion order is right)", failures)
        _check(bool(torch.isfinite(obs["policy"]).all()), "policy observation finite", failures)
        _check(bool(torch.isfinite(obs["critic"]).all()), "critic observation finite", failures)

        from luna_hifi_tasks.excavator.mdp.observations import excavate_obs_spec, navigate_obs_spec

        sp = excavate_obs_spec() if is_dig else navigate_obs_spec()
        sl = sp.slice_of("terrain_scan")
        scan = obs["policy"][:, sl]
        print(f"\n=== terrain scan ===\n  min {scan.min():.3f} max {scan.max():.3f} std {scan.std():.4f}")
        _check(float(scan.std()) > 1e-4, "terrain scan varies across cells (sensor is seeing the ground)", failures)

        if is_dig:
            print("\n=== dig: lower arms, spin drums, drive slowly ===")
            dig = torch.zeros(u.num_envs, 4, device=u.device)
            dig[:, 0] = 0.3      # forward
            dig[:, 2] = 0.9      # boom well down
            dig[:, 3] = 1.0      # drums at full dig speed
            fill0 = u._fill_kg.clone()
            for i in range(args.steps):
                obs, *_ = env.step(dig)
                if i % 50 == 0:
                    print(f"  step {i:4d}  fill kg {u._fill_kg.mean(0).tolist()}  "
                          f"arm {u.robot.data.joint_pos.torch[0, u._arm_ids].tolist()}")
            gained = (u._fill_kg - fill0).sum(dim=-1)
            print(f"  fill gained per env: {gained.tolist()}")
            _check(bool((gained > 1.0).any()), "drum fill rose with drums in the bed (particle adapter reads real particles)", failures)
        else:
            print(f"\n=== drive: {args.steps} steps forward with a slow yaw ===")
            drive = torch.zeros(u.num_envs, 2, device=u.device)
            drive[:, 0] = 0.6
            drive[:, 1] = 0.2
            x0 = u.robot.data.root_pos_w.torch[:, 0].clone()
            for _ in range(args.steps):
                obs, *_ = env.step(drive)
            dx = u.robot.data.root_pos_w.torch[:, 0] - x0
            print(f"  x displacement per env: {dx.tolist()[:8]}")
            _check(bool((dx.abs() > 0.5).any()), "machine moved under a forward command (wheel sign and drive mapping)", failures)

        _check(bool(torch.isfinite(obs["policy"]).all()), "policy observation finite after stepping", failures)
        env.close()

    print("\n=== result ===")
    if failures:
        for f in failures:
            print("  FAILED:", f)
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
