# scripts/run_task.py
#
# Build and step an excavator task with no policy.
#
#   isaaclab -p scripts/run_task.py --task Luna-Excavator-Navigate --num_envs 8
#   isaaclab -p scripts/run_task.py --task Luna-Excavator-Excavate --num_envs 2
#   isaaclab -p scripts/run_task.py --task Luna-Excavator-Navigate --watch --steps 3000
#
# Hydra overrides take Hydra syntax, no leading dashes:
#
#   isaaclab -p scripts/run_task.py --task Luna-Excavator-Navigate env.scene.num_envs=4
#
# Asserts, over the run:
#
#   observation widths match their declared specs
#   every observation is finite
#   the terrain scan is not constant
#   drum fill rises on the MPM tier with the drums buried and spinning
#   projected gravity z is near -1 after settling
#
# Prints the articulation's body and joint names, and every prim path matched
# by the coupler and ray-caster regexes.

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
    p = argparse.ArgumentParser(description="Build and step an excavator task with no policy.")
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
                   help="excavate only; soil yield_stress in Pa, applied to the MPM "
                        "material as well as the cfg field")
    p.add_argument("--terrain_rows", type=int, default=None, help="navigate only; rows of sub-terrain")
    p.add_argument("--terrain_cols", type=int, default=None, help="navigate only; cols of sub-terrain")
    add_launcher_args(p)
    # Kitless default, matching Isaac Lab's own checkpoint-free agents.
    p.set_defaults(device=None, visualizer=["newton_gl"])

    # resolve_task_config runs Hydra, which re-reads sys.argv, so this
    # script's own flags are split off first and the remainder handed over.
    # The same two lines isaaclab_rl's simple_agents uses.
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

    # Checked before anything spawns.
    from luna_hifi_tasks.excavator.excavator_cfg import usd_status

    problem = usd_status()
    if problem is not None:
        print(f"\n[asset] {problem}\n")
        return 2

    # sys.argv now holds only the Hydra remainder, so plain Hydra overrides
    # work here too, e.g.  env.scene.num_envs=4  (no leading dashes).
    env_cfg, _ = resolve_task_config(args.task, "")

    # Plain argparse rather than the Hydra CLI, so anything adjustable is an
    # explicit flag applied to the config object here.
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    elif args.watch:
        env_cfg.scene.num_envs = 1

    if args.cohesion is not None:
        env_cfg.soil_cohesion = args.cohesion
    gen = getattr(getattr(env_cfg.scene, "terrain", None), "terrain_generator", None)
    if gen is not None:
        rows = args.terrain_rows if args.terrain_rows is not None else (1 if args.watch else None)
        cols = args.terrain_cols if args.terrain_cols is not None else (1 if args.watch else None)
        if rows is not None:
            gen.num_rows = rows
        if cols is not None:
            gen.num_cols = cols
        # max_init_terrain_level is clamped to num_rows - 1 internally; this
        # keeps the config reading the same as what runs.
        env_cfg.scene.terrain.max_init_terrain_level = min(
            env_cfg.scene.terrain.max_init_terrain_level, gen.num_rows - 1
        )
        print(f"[smoke] terrain {gen.num_rows} x {gen.num_cols} sub-terrains of {gen.size} m")

    if args.watch:
        # visualizer_cfgs defaults to an empty list and --visualizer filters
        # what the config declares, so a viewer is added here.
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        show_particles = "Excavate" in args.task
        if show_particles:
            # Aimed at the bed, which sits metres ahead of the machine's
            # start at the origin.
            bed_cx = 0.5 * (env_cfg.bed_x[0] + env_cfg.bed_x[1])
            lookat = (0.5 * bed_cx, 0.0, 0.3)
            eye = (lookat[0] + 4.0, 4.0, 2.5)
        else:
            lookat, eye = (0.0, 0.0, 0.5), (7.0, 7.0, 4.5)
        env_cfg.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                show_particles=show_particles,
                particle_color=(0.62, 0.55, 0.45) if show_particles else None,
                eye=eye,
                lookat=lookat,
            )
        ]
        print("[smoke] watch mode: Newton GL window, "
              f"{env_cfg.scene.num_envs} env, particles={show_particles}, "
              f"eye {eye} -> {lookat}")

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
        print("  shroud ids", u._shroud_ids)
        if is_dig:
            # Which prims the MPM coupling regex matched. The shroud hangs
            # off the arm rather than the drum, one level out from the
            # rotor.
            import re as _re

            pattern = _re.compile(u.cfg.soil_contact_regex)
            try:
                from pxr import Usd  # noqa: F401

                stage = u.sim.stage if hasattr(u.sim, "stage") else None
                paths = ([p.GetPath().pathString for p in stage.Traverse()] if stage else [])
            except Exception:
                paths = []
            hits = [p for p in paths if pattern.match(p)]
            if hits:
                print(f"  soil_contact_regex matched {len(hits)} prims:")
                for h in hits[:16]:
                    print("   ", h)
            else:
                print("  (could not list prims here; the coupler's own body count is")
                print("   the fallback -- 4 wheels + 2 rotors + 2 shrouds should be 8)")
        else:
            # The rigid tier has no coupler; its sensors are the two ray
            # casters. GridPatternCfg puts a ray at both ends of each axis, so
            # the ray counts are what confirm the (NX-1)*cell sizes.
            from luna_hifi_tasks.excavator.mdp.observations import (
                NAV_FAR_CELLS, NAV_NEAR_CELLS,
            )

            for name, want in (("far_scanner", NAV_FAR_CELLS), ("near_scanner", NAV_NEAR_CELLS)):
                rays = u.scene[name].data.ray_hits_w.torch.shape[1]
                _check(rays == want, f"{name} casts {rays} rays == {want} declared cells", failures)
            levels = u.terrain.terrain_levels
            print(f"  terrain levels: {int(levels.min())}-{int(levels.max())} of "
                  f"{int(u.terrain.max_terrain_level)}, mean {levels.float().mean():.2f}")

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
        names = ["terrain_scan"] if is_dig else ["far_scan", "near_scan"]
        print("\n=== terrain scan ===")
        for name in names:
            scan = obs["policy"][:, sp.slice_of(name)]
            print(f"  {name:<11} min {scan.min():.3f} max {scan.max():.3f} std {scan.std():.4f}")
            _check(float(scan.std()) > 1e-4,
                   f"{name} varies across cells (sensor is seeing the ground)", failures)

        if is_dig:
            print("\n=== dig: lower arms, spin drums, drive slowly ===")
            from luna_hifi_tasks.excavator.excavator import DIG_DRUM_SIGN

            dig = torch.zeros(u.num_envs, u.cfg.action_space, device=u.device)
            _, shi = u.cfg.shroud_range
            # Solved from the scene: the command that reaches a given depth
            # depends on the bed and on whether the machine stands on it.
            cut = 0.08
            boom, arm = u.cfg.boom_command_for_cut(cut)
            dig[:, 0] = 0.05                    # 0.075 m/s, the crawl dig_demo uses
            dig[:, 2] = boom
            dig[:, 3] = DIG_DRUM_SIGN           # rotors at the loading sign
            # Inlet held at minus the arm angle, so it points at the ground.
            dig[:, 4] = max(-1.0, min(1.0, -arm / max(shi, 1e-6)))
            print(f"  cut {cut:.2f} m -> boom {boom:+.3f}, arm {arm:.3f} rad, "
                  f"crawl {0.05 * 5.0 * 0.30:.3f} m/s over {args.steps} steps")

            def _where(tag: str) -> None:
                """Drum, soil and drive state, all in the env frame.

                Separates the three things a zero fill reading can mean: the
                drum is not in the soil, the soil is not where the cfg says,
                or the machine never moved.
                """
                from luna_hifi_tasks.excavator.excavate.excavate_env_cfg import (
                    BORE_HALF_LEN, BORE_RADIUS,
                )
                from luna_hifi_tasks.excavator.mdp.sensors import drum_fill_mass

                org = u.scene.env_origins[0]
                pos, env = u._particles()
                p0 = pos[env == 0] - org
                dpos, dquat = u._drum_poses()
                d = dpos[0, 0] - org
                chassis = u.robot.data.root_pos_w.torch[0] - org
                # The fill sensor's own test, in the drum frame, at the bore
                # and at a shell 0.15 m wider. Soil in the shell but not the
                # bore means the drum is beside it.
                def mass(radius: float) -> float:
                    return float(drum_fill_mass(
                        pos, env, u._particle_mass, dpos[:, 0], dquat[:, 0],
                        radius, BORE_HALF_LEN, u.num_envs,
                    )[0])
                inside, near = mass(BORE_RADIUS), mass(BORE_RADIUS + 0.15)
                wv = u.robot.data.joint_vel.torch[0, u._wheel_ids]
                tau = u.robot.data.applied_torque.torch[0, u._wheel_ids]
                print(f"  [{tag}] chassis x={chassis[0]:+.3f} z={chassis[2]:.3f}   "
                      f"drum x={d[0]:.3f} z={d[2]:.3f}  shroud bottom z={d[2] - 0.212:.3f}")
                c = u.cfg
                fm = c.mpm_floor_margin
                fx = (c.bed_x[0] - fm, c.bed_x[1] + fm)
                fy = (c.bed_y[0] - fm, c.bed_y[1] + fm)
                off = int((
                    (p0[:, 0] < fx[0]) | (p0[:, 0] > fx[1])
                    | (p0[:, 1] < fy[0]) | (p0[:, 1] > fy[1])
                ).sum())
                below = int((p0[:, 2] < -0.10).sum())
                print(f"  [{tag}] soil {p0.shape[0]} particles  "
                      f"x[{p0[:, 0].min():.2f},{p0[:, 0].max():.2f}] "
                      f"y[{p0[:, 1].min():+.2f},{p0[:, 1].max():+.2f}] "
                      f"z[{p0[:, 2].min():.3f},{p0[:, 2].max():.3f}]   "
                      f"bed x{tuple(c.bed_x)} y{tuple(c.bed_y)} top {c.bed_top:.3f}")
                print(f"  [{tag}] past the MPM floor x{fx} y{fy}: {off} particles, "
                      f"{below} below it (those have nothing to stand on)")
                print(f"  [{tag}] soil in the bore {inside:.2f} kg, within +0.15 m {near:.2f} kg   "
                      f"wheel vel {wv.mean():+.3f} rad/s (cmd {u._forward_cmd[0] / 0.30:+.3f})  "
                      f"torque {tau.abs().mean():.1f} N-m")

            _where("start")
            peak = u._fill_kg.sum(dim=-1).clone()
            for i in range(args.steps):
                obs, *_ = env.step(dig)
                peak = torch.maximum(peak, u._fill_kg.sum(dim=-1))
                if i % 50 == 0:
                    print(f"  step {i:4d}  fill kg {u._fill_kg.mean(0).tolist()}  "
                          f"arm {u.robot.data.joint_pos.torch[0, u._arm_ids].tolist()}")
            # Peak over the run, since a machine that crosses the bed and
            # carries on sheds its load.
            _where("end")
            print(f"  peak fill per env: {peak.tolist()}")
            _check(bool((peak > 1.0).any()), "drum fill rose with drums in the bed (particle adapter reads real particles)", failures)
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
