# scripts/smoke_test.py
#
# Run this before the first training job, on a machine with Isaac Lab:
#
#   isaaclab -p scripts/smoke_test.py --task Luna-Excavator-Navigate --num_envs 8
#   isaaclab -p scripts/smoke_test.py --task Luna-Excavator-Excavate --num_envs 2
#
# Add --watch to open a Newton GL window and follow one machine:
#
#   isaaclab -p scripts/smoke_test.py --task Luna-Excavator-Navigate --watch --steps 3000
#
# Hydra overrides also work, in Hydra syntax (no leading dashes):
#
#   isaaclab -p scripts/smoke_test.py --task Luna-Excavator-Navigate env.scene.num_envs=4
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
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)


def _parse(argv):
    p = argparse.ArgumentParser(description="Excavator environment smoke test.")
    p.add_argument("--task", required=True, help="Luna-Excavator-Navigate or Luna-Excavator-Excavate")
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument(
        "--watch",
        action="store_true",
        help="Open a Newton GL window. Implies one environment and the smallest "
             "terrain, unless --num_envs / --terrain_rows / --terrain_cols say otherwise.",
    )
    p.add_argument("--cohesion", type=float, default=None,
                   help="excavate only; soil yield_stress in Pa. An explicit flag because a "
                        "Hydra override lands after the MPM material is built from the field")
    p.add_argument("--terrain_rows", type=int, default=None, help="navigate only; rows of sub-terrain")
    p.add_argument("--terrain_cols", type=int, default=None, help="navigate only; cols of sub-terrain")
    add_launcher_args(p)
    # Kitless default, matching Isaac Lab's own checkpoint-free agents.
    p.set_defaults(device=None, visualizer=["newton_gl"])

    # resolve_task_config runs Hydra, and Hydra re-reads sys.argv. So this
    # script's own flags have to be split off first and the REMAINDER handed
    # over, or Hydra rejects --task and friends as unrecognised. Same two lines
    # isaaclab_rl's simple_agents uses, for the same reason.
    args, hydra_args = setup_preset_cli(p, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def _check(cond: bool, msg: str, failures: list[str]) -> None:
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def main(argv=None) -> int:
    args = _parse(argv)
    torch.manual_seed(0)

    # sys.argv now holds only the Hydra remainder, so plain Hydra overrides
    # work here too, e.g.  env.scene.num_envs=4  (no leading dashes).
    env_cfg, _ = resolve_task_config(args.task, "")

    # This script uses plain argparse, not the Hydra CLI, so `--env.*` overrides
    # do NOT reach it. Anything adjustable has to be an explicit flag and get
    # applied to the config object here.
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    elif args.watch:
        env_cfg.scene.num_envs = 1

    if args.cohesion is not None:
        env_cfg.soil_cohesion = args.cohesion
    if hasattr(env_cfg, "apply_soil_material"):
        # Also repairs a Hydra override that landed on a soil_* field after
        # __post_init__ had already copied it onto the material.
        env_cfg.apply_soil_material()

    gen = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
    if gen is not None:
        rows = args.terrain_rows if args.terrain_rows is not None else (1 if args.watch else None)
        cols = args.terrain_cols if args.terrain_cols is not None else (1 if args.watch else None)
        if rows is not None:
            gen.num_rows = rows
        if cols is not None:
            gen.num_cols = cols
        # max_init_terrain_level is clamped to num_rows - 1 internally, but keep
        # the config self-consistent so it reads honestly in the log.
        env_cfg.scene.terrain.max_init_terrain_level = min(
            env_cfg.scene.terrain.max_init_terrain_level, gen.num_rows - 1
        )
        print(f"[smoke] terrain {gen.num_rows} x {gen.num_cols} sub-terrains of {gen.size} m")

    if args.watch:
        # visualizer_cfgs defaults to an empty list and the --visualizer flag
        # only filters what the config already declares, so a viewer has to be
        # added here. play_mode() would do it, but nothing calls play_mode on
        # this path.
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        show_particles = "Excavate" in args.task
        env_cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                show_particles=show_particles,
                particle_color=(0.62, 0.55, 0.45) if show_particles else None,
                eye=(7.0, 7.0, 4.5),
                lookat=(0.0, 0.0, 0.5),
            )
        ]
        print("[smoke] watch mode: Newton GL window, "
              f"{env_cfg.scene.num_envs} env, particles={show_particles}")

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
