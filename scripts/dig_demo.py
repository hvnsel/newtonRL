# scripts/dig_demo.py
#
# Scripted pass on the small bed, no policy. The env settles the bed, lowers
# the arms and spins the drums up (dig_cut, dig_drum_speed, dig_ramp_seconds);
# then the machine crawls forward at --drive.
#
#   isaaclab -p scripts/dig_demo.py
#   isaaclab -p scripts/dig_demo.py --cut 0.05 --drive 0.2 --seconds 20
#
# Exits 0 if drum fill rose above 1 kg in every env. --no_window skips the viewer.

from __future__ import annotations

import argparse
import math
import sys
import time

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)
from luna_hifi_tasks.excavator.excavator import (
    ARM_LEN,
    DIG_DRUM_SIGN,
    PIVOT_X,
    ROTOR_TIP_R,
    SHROUD_OUT_R,
    VANE_TIP_GAP,
    WHEEL_RADIUS,
)
from luna_hifi_tasks.excavator.excavator_cfg import (
    MAX_DRUM_SPEED,
    MAX_WHEEL_SPEED,
    SPIN_LIMIT,
    usd_status,
)
from luna_hifi_tasks.excavator.excavate.excavate_env_cfg import BORE_HALF_LEN
from luna_hifi_tasks.excavator.mdp.sensors import drum_fill_mass

TASK = "Luna-Excavator-Excavate-Small"
CHANNEL_OPENING = VANE_TIP_GAP


def _parse(argv):
    p = argparse.ArgumentParser(description="Scripted dig on the small regolith bed.")
    p.add_argument("--task", default=TASK, help=f"default {TASK}")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--seconds", type=float, default=30.0, help="simulated seconds to run")
    p.add_argument("--no_window", action="store_true", help="skip the viewer; just run the checks")

    p.add_argument("--cut", type=float, default=None,
                   help="override dig_cut: how deep the shroud bottom goes below the surface, metres")
    p.add_argument("--drum", type=float, default=None,
                   help="override dig_drum_speed as a fraction of MAX_DRUM_SPEED. "
                        f"The loading sign is {DIG_DRUM_SIGN:+.0f}; the other dumps")
    p.add_argument("--drive", type=float, default=0.05,
                   help="forward command once the tool is down, a fraction of 1.5 m/s; at most 0.30")

    # Set on the cfg before gym.make, which re-derives the MPM material from them.
    p.add_argument("--cohesion", type=float, default=None, help="soil yield_stress in Pa")
    p.add_argument("--friction", type=float, default=None, help="soil friction, ~tan(phi)")
    p.add_argument("--density", type=float, default=None, help="soil density, kg/m3")
    p.add_argument("--voxel", type=float, default=None,
                   help="override voxel_size [m], the height-grid cell; half of it is the collider margin")

    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=["newton_gl"])
    # Hydra (in resolve_task_config) reads the remaining arguments from sys.argv.
    args, hydra_args = setup_preset_cli(p, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def _dominant(u, n: int = 2) -> str:
    """The n reward terms with the largest magnitude this step, for env 0."""
    ranked = sorted(
        ((k, float(v[0])) for k, v in u._last_terms.items()),
        key=lambda kv: -abs(kv[1]),
    )[:n]
    return "[" + " ".join(f"{k} {v:+.2f}" for k, v in ranked if abs(v) > 1e-3) + "]"


def soil_shells(u) -> tuple[torch.Tensor, torch.Tensor]:
    """(E, 2) kg per drum, front and rear, in two radial bands about the drum
    axis: ROTOR_TIP_R..SHROUD_OUT_R (between vane tips and shroud wall), and
    SHROUD_OUT_R..SHROUD_OUT_R + 0.10 (outside the shroud)."""
    pos, env = u._particles()
    dpos, dquat = u._drum_poses()

    def within(radius: float) -> torch.Tensor:
        return torch.stack([
            drum_fill_mass(pos, env, u._particle_mass, dpos[:, i], dquat[:, i],
                           radius, BORE_HALF_LEN, u.num_envs)
            for i in range(2)
        ], dim=-1)

    shell = (within(SHROUD_OUT_R) - within(ROTOR_TIP_R)).clamp(min=0.0)
    outside = (within(SHROUD_OUT_R + 0.10) - within(SHROUD_OUT_R)).clamp(min=0.0)
    return shell, outside


def cut_diagnosis(cfg, cut: float) -> list[str]:
    """Notes on a shroud cut depth `cut` against the bed: whether the shroud
    reaches the floor, and how much of the rotor ends up below grade."""
    wall = SHROUD_OUT_R - ROTOR_TIP_R
    floor = cfg.bed_top - cfg.bed_depth
    shroud_z = cfg.bed_top - cut

    out = []
    if shroud_z < floor:
        out.append(f"!! the SHROUD would sit {floor - shroud_z:.3f} m below the bed floor "
                   f"at z={floor:.3f}: it grinds on the slab, it does not dig")
    if cut > cfg.bed_depth:
        out.append(f"deepest cut this {cfg.bed_depth:.2f} m bed supports is "
                   f"{cfg.bed_depth:.3f} m; try --cut {max(cfg.bed_depth - 0.005, 0.0):.2f}")

    span = 2.0 * ROTOR_TIP_R
    submerged = max(min(cut - wall, span), 0.0)
    out.append(f"{submerged / span * 100:.0f}% of the {span:.2f} m rotor "
               f"ends up below grade")
    if submerged < ROTOR_TIP_R:
        out.append(f"under half; a {ROTOR_TIP_R + wall:.2f} m cut buries half the rotor")
    return out


def main(argv=None) -> int:
    args = _parse(argv)
    torch.manual_seed(0)

    problem = usd_status()
    if problem is not None:
        print(f"\n[asset] {problem}\n")
        return 2

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs

    if args.voxel is not None:
        env_cfg.voxel_size = args.voxel
    for flag, field in (("cohesion", "soil_cohesion"),
                        ("friction", "soil_friction"),
                        ("density", "soil_density")):
        value = getattr(args, flag)
        if value is not None:
            setattr(env_cfg, field, value)
    if args.cut is not None:
        env_cfg.dig_cut = args.cut
    if args.drum is not None:
        env_cfg.dig_drum_speed = args.drum * MAX_DRUM_SPEED

    if not args.no_window:
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
        print(f"  drum interior holds {u.cfg.drum_capacity_kg:.1f} kg each")
        print("  shroud inlet held at minus the arm angle, so it stays "
              "pointed at the ground")
        print(f"  machine stands {'ON the bed' if u.cfg.spawn_on_bed else 'beside the pile'}")
        mat = u.cfg.scene.soil.spawn.material
        print(f"  soil: density {mat.density:.0f} kg/m3, friction {mat.friction:.2f}, "
              f"cohesion {mat.yield_stress:.0f} Pa")
        # Entry channel: the vane-tip gap less 6 mm a side. Width in voxels after
        # one voxel of collider margin.
        chan = CHANNEL_OPENING - 2.0 * 0.006
        grains = (chan - u.cfg.voxel_size) / u.cfg.voxel_size
        print(f"  entry channel {chan:.4f} m, {u.cfg.voxel_size:.3f} m voxel -> "
              f"{grains:.1f} grains wide "
              f"({'arches' if grains < 3.0 else 'marginal' if grains < 4.5 else 'flows'})")
        rad_s = u.cfg.dig_drum_speed
        print(f"  drum: {rad_s:+.1f} rad/s -> tip {abs(rad_s) * ROTOR_TIP_R:.2f} m/s "
              f"(at r = {ROTOR_TIP_R:.3f} m, the vane tip)")
        if abs(rad_s) > SPIN_LIMIT:
            print(f"  !! {abs(rad_s):.2f} rad/s is over the {SPIN_LIMIT:.2f} rad/s where "
                  "outward acceleration passes lunar gravity")
        if rad_s * DIG_DRUM_SIGN < 0:
            print(f"  * this is the DUMP direction. Loading is {DIG_DRUM_SIGN:+.0f}")

        cut = u.cfg.dig_cut
        angle = u._dig_arm

        drum_x = PIVOT_X + ARM_LEN * math.cos(angle)
        lead = drum_x + SHROUD_OUT_R
        ahead = u.cfg.bed_x[1] - lead
        speed = args.drive * MAX_WHEEL_SPEED * WHEEL_RADIUS
        print("\n=== drum against the bed ===")
        print(f"  bed spans x {u.cfg.bed_x[0]:.2f} .. {u.cfg.bed_x[1]:.2f} m; at the dig "
              f"angle the drum axis is at {drum_x:.2f}, leading edge {lead:.2f}")
        if ahead <= 0.0:
            print(f"  !! the drum is already {-ahead:.2f} m PAST the far edge. Move bed_x "
                  f"forward or it cannot pick anything up")
        else:
            print(f"  {ahead:.2f} m of bed ahead of it; crawling {speed:.3f} m/s "
                  f"clears it in {ahead / max(speed, 1e-6):.1f} s")
        wall = SHROUD_OUT_R - ROTOR_TIP_R
        print(f"  to cut {cut:.3f} m -> arm {angle:+.3f} rad, shroud {u._dig_shroud:+.3f} rad")
        print(f"  vane tips reach {cut - wall:.3f} m below grade")
        for line in cut_diagnosis(u.cfg, cut):
            print(f"  * {line}")
        print()

        obs, _ = env.reset()
        # [forward, yaw]: forward is dig_forward * (1 + a0) once the tool is down.
        action = torch.zeros(u.num_envs, 2, device=u.device)
        action[:, 0] = args.drive / u.cfg.dig_forward - 1.0
        t_start = time.perf_counter()

        peak = torch.zeros(u.num_envs, 2, device=u.device)
        peak_lip = torch.zeros(u.num_envs, 2, device=u.device)
        phase = ""
        for i in range(steps):
            t = i * dt
            since = int(u._since_settle[0])
            now = ("settle" if bool(u._cut_fresh[0]) else
                   "lower and spin" if since < u._ramp_steps else "drive")
            if now != phase:
                phase = now
                print(f"[{t:5.1f}s] phase: {phase}")

            obs, rew, term, trunc, _ = env.step(action)
            peak = torch.maximum(peak, u._fill_kg)

            if i % int(0.5 / dt) == 0:
                arm = float(u.robot.data.joint_pos.torch[0, u._arm_ids[0]])
                fill = u._fill_kg[0]
                wall, out = soil_shells(u)
                peak_lip = torch.maximum(peak_lip, wall)
                print(f"  t={t:5.1f}s  arm={arm:+.3f} rad  "
                      f"bore=[{fill[0]:6.2f},{fill[1]:6.2f}]  "
                      f"wall=[{wall[0][0]:5.2f},{wall[0][1]:5.2f}]  "
                      f"buried-in={out[0][0]:6.1f}  "
                      f"rew={float(rew[0]):+6.3f}  {_dominant(u)}")
            if bool(term.any()) or bool(trunc.any()):
                print(f"  t={t:5.1f}s  episode ended (terminated={bool(term.any())}, "
                      f"truncated={bool(trunc.any())}) -- bed and machine reset")

        wall = time.perf_counter() - t_start
        sps = steps * u.num_envs / wall
        print("\n=== throughput ===")
        print(f"  {steps} steps x {u.num_envs} env in {wall:.1f} s "
              f"-> {sps:,.0f} env-steps/s at {u.cfg.bed_particles_per_env:,} particles")

        print("\n=== result ===")
        print(f"  peak fill per drum, per env (kg):\n{peak}")
        print(f"  peak in the shell wall band, never reaching the bore (kg):\n{peak_lip}")
        ok = bool((peak.sum(dim=-1) > 1.0).all())
        print(("  PASS  " if ok else "  FAIL  ") +
              "drum fill rose while the drums were in the bed")
        if not ok:
            print("  check the arm angle above against bed_top (--cut), the body names")
            print("  against SOIL_CONTACT_BODIES_REGEX, the entry channel width (--voxel) and")
            print(f"  the drum sign (--drum; loading is {DIG_DRUM_SIGN:+.0f})")
        env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
