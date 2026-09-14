# scripts/dig_demo.py
#
# Watch the machine dig, with no policy involved. A fixed four-phase script
# drives the excavate task on the small bed:
#
#   settle   everything at rest, arms stowed high, so the chassis finds the
#            soil surface before anything moves
#   lower    boom ramps down until the drums are buried to the target depth
#   spin     drums ramp up to digging speed, still stationary
#   drive    slow forward crawl, cutting a trench
#
#   isaaclab -p scripts/dig_demo.py
#   isaaclab -p scripts/dig_demo.py --boom 0.9 --drive 0.35 --seconds 40
#
# The point is not the motion -- it is the FILL READOUT. drum_fill_mass is the
# entire excavation reward, and it reads particle positions through an adapter
# that was written against the isaaclab_newton source without ever being run.
# If that adapter is wrong it returns zero, which during training is
# indistinguishable from a policy that has not learned to dig. Here the drums
# are provably in the soil, so fill MUST rise. It is the one check that cannot
# be made without a simulator.
#
# Pass --headless to run it as a plain assertion with no window.

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

TASK = "Luna-Excavator-Excavate-Small"


def _parse(argv):
    p = argparse.ArgumentParser(description="Scripted dig on the small regolith bed.")
    p.add_argument("--task", default=TASK, help=f"default {TASK}")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--seconds", type=float, default=30.0, help="simulated seconds to run")
    p.add_argument("--headless", action="store_true", help="no window; just run the checks")

    # Phase boundaries, in simulated seconds.
    p.add_argument("--t_lower", type=float, default=1.5, help="start lowering the boom")
    p.add_argument("--t_spin", type=float, default=5.0, help="start the drums")
    p.add_argument("--t_drive", type=float, default=7.0, help="start crawling forward")

    # Commands, all in the env's [-1, 1] action units.
    #
    # boom maps to an arm angle through arm_range = (-0.55, 0.80):
    #   angle = lo + 0.5 * (a + 1) * (hi - lo)
    # and the arm angle decides how deep the drum sits. On this bed the wheels
    # rest ON the 0.15 m soil surface, so the drum has a long way to reach
    # before it touches anything:
    #
    #   cmd    angle      drum bottom      cuts
    #   0.00   +0.125     z = +0.390       nothing, 0.2 m clear of the surface
    #   0.40   +0.395     z = +0.213       nothing, still above it
    #   0.60   +0.530     z = +0.131       0.044 m
    #   0.78   +0.651     z = +0.063       0.112 m   <- default
    #   0.90   +0.732     z = +0.020       through the bed into the floor slab
    #
    # So the useful band is narrow, roughly 0.6 to 0.85. Below it the drum
    # waves in the air and fill stays at zero for a reason that has nothing to
    # do with the sensor; above it the drum grinds on the hidden slab.
    p.add_argument("--boom", type=float, default=0.78,
                   help="boom command once lowered; useful range ~0.6-0.85 on this bed")
    p.add_argument("--drum", type=float, default=1.0, help="drum command once spinning")
    p.add_argument("--drive", type=float, default=0.25, help="forward command once crawling")

    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=["newton_gl"])
    # resolve_task_config runs Hydra, which re-reads sys.argv, so this script's
    # own flags have to be split off and the remainder handed over.
    args, hydra_args = setup_preset_cli(p, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def _ramp(t: float, t0: float, duration: float = 1.5) -> float:
    """0 before t0, 1 after t0 + duration, linear between. Ramping rather than
    stepping keeps the arm from slamming into the bed and the drums from
    spiking the coupler on their first step."""
    if t <= t0:
        return 0.0
    return min((t - t0) / duration, 1.0)


def main(argv=None) -> int:
    args = _parse(argv)
    torch.manual_seed(0)

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs

    if not args.headless:
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

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        u = env.unwrapped
        dt = u.step_dt
        steps = int(args.seconds / dt)

        print("\n=== articulation ===")
        print("  bodies:", u.robot.body_names)
        print("  joints:", u.robot.joint_names)
        print(f"\n=== bed ===\n  {u.cfg.bed_particles_per_env} particles/env, "
              f"grid {u.cfg.bed_grid_nx} x {u.cfg.bed_grid_ny}, "
              f"surface at z = {u.cfg.bed_top:.3f} m")
        print(f"  drum capacity {u.cfg.drum_capacity_kg:.1f} kg each\n")

        obs, _ = env.reset()
        action = torch.zeros(u.num_envs, 4, device=u.device)

        peak = torch.zeros(u.num_envs, 2, device=u.device)
        phase = ""
        for i in range(steps):
            t = i * dt

            lower = _ramp(t, args.t_lower, 2.0)
            spin = _ramp(t, args.t_spin, 1.0)
            drive = _ramp(t, args.t_drive, 1.5)

            now = ("settle" if t < args.t_lower else
                   "lower" if t < args.t_spin else
                   "spin" if t < args.t_drive else "drive")
            if now != phase:
                phase = now
                print(f"[{t:5.1f}s] phase: {phase}")

            action[:, 0] = args.drive * drive
            action[:, 1] = 0.0
            # Boom starts at the cfg's stow angle and ramps to the dig command.
            action[:, 2] = -1.0 + (args.boom + 1.0) * lower
            action[:, 3] = args.drum * spin

            obs, rew, term, trunc, _ = env.step(action)
            peak = torch.maximum(peak, u._fill_kg)

            if i % int(0.5 / dt) == 0:
                arm = float(u.robot.data.joint_pos.torch[0, u._arm_ids[0]])
                fill = u._fill_kg[0]
                print(f"  t={t:5.1f}s  arm={arm:+.3f} rad  "
                      f"fill=[{fill[0]:7.2f}, {fill[1]:7.2f}] kg  "
                      f"reward={float(rew[0]):+7.3f}")
            if bool(term.any()) or bool(trunc.any()):
                print(f"  t={t:5.1f}s  episode ended (terminated={bool(term.any())}, "
                      f"truncated={bool(trunc.any())}) -- bed and machine reset")

        print("\n=== result ===")
        print(f"  peak fill per drum, per env (kg):\n{peak}")
        ok = bool((peak.sum(dim=-1) > 1.0).all())
        print(("  PASS  " if ok else "  FAIL  ") +
              "drum fill rose while the drums were in the bed")
        if not ok:
            print("\n  Fill never moved. In order of likelihood:")
            print("   1. mpm_particle_state reads the wrong MPMObject.data attribute,")
            print("      so drum_fill_mass is summing over an empty tensor.")
            print("   2. the drums never actually reached soil -- check the arm angle")
            print("      printed above against bed_top, and try a larger --boom.")
            print("   3. the coupler proxy body regex matched nothing, so the drums")
            print("      pass through the particles without disturbing them. Compare")
            print("      the body names above with SOIL_CONTACT_BODIES_REGEX.")
        env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
