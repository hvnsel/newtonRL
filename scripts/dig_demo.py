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
#   isaaclab -p scripts/dig_demo.py --cut 0.05 --drive 0.35 --seconds 40
#
# The output that matters is the FILL READOUT. drum_fill_mass is the entire
# excavation reward; a broken particle adapter returns zero, which in training
# looks exactly like a policy that has not learned to dig. Here the drums are
# provably in the soil, so fill must rise.
#
# Pass --no_window to run it as a plain assertion with no viewer.

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
    CHASSIS_Z,
    MAST_TOP_Z,
    ROTOR_TIP_R,
    SHROUD_OUT_R,
    VANE_TIP_GAP,
)
from luna_hifi_tasks.excavator.excavator_cfg import MAX_DRUM_SPEED, SPIN_LIMIT

CHANNEL_OPENING = VANE_TIP_GAP
from luna_hifi_tasks.excavator.excavate.excavate_env_cfg import BORE_HALF_LEN
from luna_hifi_tasks.excavator.mdp.sensors import drum_fill_mass

TASK = "Luna-Excavator-Excavate-Small"


def _parse(argv):
    p = argparse.ArgumentParser(description="Scripted dig on the small regolith bed.")
    p.add_argument("--task", default=TASK, help=f"default {TASK}")
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--seconds", type=float, default=30.0, help="simulated seconds to run")
    # NOT --headless: AppLauncher guards that name as a SimulationApp config
    # key (it never adds the flag, but _check_argparser_config_params rejects a
    # parser that already carries it).
    p.add_argument("--no_window", action="store_true", help="skip the viewer; just run the checks")

    # Phase boundaries, in simulated seconds.
    p.add_argument("--t_lower", type=float, default=1.5, help="start lowering the boom")
    p.add_argument("--t_spin", type=float, default=5.0, help="start the drums")
    p.add_argument("--t_drive", type=float, default=7.0, help="start crawling forward")

    # Depth of the SHELL below the soil surface, in metres, not a boom command:
    # the command reaching a given depth depends on the bed depth and on
    # whether the machine stands on the soil or beside it, and
    # boom_command_for_cut() solves it from the env's own config.
    #
    # Measured to the SHROUD, which is the outermost thing on the drum and the
    # first to touch anything. cut_diagnosis() reports the margin per run.
    p.add_argument("--cut", type=float, default=0.10,
                   help="how deep the SHROUD should cut below the surface, metres")
    p.add_argument("--boom", type=float, default=None,
                   help="raw boom command, overriding --cut")
    # 1.0 is 8 rad/s, 2.0 m/s at the lip, which throws soil clear of the drum;
    # 0.4 is 0.8 m/s and scoops. DIG_DRUM_SIGN is the loading sign, derived
    # from the lip handedness; the other sign dumps.
    p.add_argument("--drum", type=float, default=0.4 * DIG_DRUM_SIGN,
                   help="drum command once spinning; |cmd| > ~0.5 flings rather than scoops. "
                        f"The loading sign for this lip is {DIG_DRUM_SIGN:+.0f}")
    p.add_argument("--drive", type=float, default=0.25, help="forward command once crawling")

    # Explicit flags rather than Hydra overrides: `env.soil_cohesion=1500` sets
    # the cfg field, but the MPM material was built from it in __post_init__,
    # which Hydra runs after. These set the field and re-run
    # apply_soil_material().
    p.add_argument("--cohesion", type=float, default=None,
                   help="soil yield_stress in Pa; 0 sprays, ~800 holds a cut together")
    p.add_argument("--friction", type=float, default=None, help="soil friction, ~tan(phi)")
    p.add_argument("--density", type=float, default=None, help="soil density, kg/m3")
    # The channel opens 0.122 m, the coupler eats a whole voxel of it and
    # particles spawn one voxel apart, so this sets how many grains wide the
    # way in is. Granular material arches across an orifice narrower than about
    # four of them however hard it is driven.
    #   0.03 -> 3.0 grains   0.025 -> 3.9   0.02 -> 5.0
    # Finer costs particles as the cube.
    p.add_argument("--voxel", type=float, default=None,
                   help="MPM voxel; sets how many grains wide the drum's entry channel is")

    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=["newton_gl"])
    # resolve_task_config runs Hydra, which re-reads sys.argv, so this script's
    # own flags have to be split off and the remainder handed over.
    args, hydra_args = setup_preset_cli(p, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def boom_command_for_cut(cfg, cut: float) -> tuple[float, float]:
    """Boom command in [-1, 1] that puts the drum `cut` metres into the soil.

    Geometry, all heights measured in world z:

        wheel plane   = bed surface if the machine stands on the bed,
                        otherwise 0 (it is beside the pile, on the ground)
        drum bottom   = wheel plane + pivot - ARM_LEN*sin(angle) - SHROUD_OUT_R
        target        = bed surface - cut

    Solve for the angle, clamp to the joint's range, then invert the linear
    action mapping the env applies:  angle = lo + 0.5*(cmd + 1)*(hi - lo).
    """
    pivot = CHASSIS_Z + MAST_TOP_Z          # pivot height above the wheel plane
    wheel_plane = cfg.bed_top if cfg.spawn_on_bed else 0.0
    target = cfg.bed_top - cut

    sin_t = (wheel_plane + pivot - SHROUD_OUT_R - target) / ARM_LEN
    angle = math.asin(min(max(sin_t, -1.0), 1.0))

    lo, hi = cfg.arm_range
    angle = min(max(angle, lo), hi)
    return 2.0 * (angle - lo) / (hi - lo) - 1.0, angle


def soil_shells(u) -> tuple[torch.Tensor, torch.Tensor]:
    """(in the running clearance, outside the shroud) kg per drum, front and rear.

    Three radial bands, because one number cannot tell the interesting states
    apart:

        r < 0.185   the rotor's swept cylinder. This is fill, and the only
                    thing that counts.
        0.185-0.212 the running clearance and the shroud wall: soil wedged
                    between vane tips and shroud rather than held in a pocket.
        > 0.212     outside the drum altogether -- soil the shroud happens to
                    be buried in. It rises whenever the drum is in the ground
                    and means nothing.
    """
    pos, env = u._particles()
    dpos, dquat = u._drum_poses()

    def within(radius: float) -> torch.Tensor:
        return torch.stack([
            drum_fill_mass(pos, env, u._particle_mass, dpos[:, i], dquat[:, i],
                           radius, BORE_HALF_LEN, u.num_envs)
            for i in range(2)
        ], dim=-1)

    shell = (within(SHROUD_OUT_R) - u._fill_kg).clamp(min=0.0)
    outside = (within(SHROUD_OUT_R + 0.10) - within(SHROUD_OUT_R)).clamp(min=0.0)
    return shell, outside


def cut_diagnosis(cfg, cut: float) -> list[str]:
    """Whether the bed can actually supply the requested cut.

    The bed is a finite slab on a rigid floor, so `cut` is not free: past a
    point the drum is no longer cutting soil, it is grinding on the hidden slab
    that holds the particles up, and the run looks like a dig that will not
    load. `cut` refers to the SHROUD, which is the outermost thing on the drum
    and the first to touch anything.
    """
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
        out.append(f"that is under half. A {ROTOR_TIP_R + wall:.2f} m deep bed is what it "
                   "takes to bury half the rotor, and no cut depth or cohesion value "
                   "substitutes for it")
    return out


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

    # Before anything spawns: a missing or stale asset otherwise surfaces as a
    # FileNotFoundError deep inside InteractiveScene, or -- worse -- as a
    # perfectly clean run of the PREVIOUS drum.
    from luna_hifi_tasks.excavator.excavator_cfg import usd_status

    problem = usd_status()
    if problem is not None:
        print(f"\n[asset] {problem}\n")
        return 2

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs

    if args.voxel is not None:
        env_cfg.voxel_size = args.voxel
        # Re-derives the bed, the particle count and the grid caps from it.
        env_cfg.__post_init__()
    for flag, field in (("cohesion", "soil_cohesion"),
                        ("friction", "soil_friction"),
                        ("density", "soil_density")):
        value = getattr(args, flag)
        if value is not None:
            setattr(env_cfg, field, value)
    # Unconditional: this both applies the flags above and repairs any Hydra
    # override that landed on a soil_* field after __post_init__ had already
    # built the material from it. The env raises if the two still disagree.
    env_cfg.apply_soil_material()

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
        print(f"  drum capacity {u.cfg.drum_capacity_kg:.1f} kg each")
        print(f"  machine stands {'ON the bed' if u.cfg.spawn_on_bed else 'beside the pile'}")
        # Off the material the solver was handed, not off the cfg fields.
        mat = u.cfg.scene.soil.spawn.material
        print(f"  soil: density {mat.density:.0f} kg/m3, friction {mat.friction:.2f}, "
              f"cohesion {mat.yield_stress:.0f} Pa")
        # The number that decides whether soil can enter at all.
        chan = CHANNEL_OPENING - 2.0 * 0.006
        grains = (chan - u.cfg.voxel_size) / u.cfg.voxel_size
        print(f"  entry channel {chan:.4f} m, {u.cfg.voxel_size:.3f} m voxel -> "
              f"{grains:.1f} grains wide "
              f"({'arches' if grains < 3.0 else 'marginal' if grains < 4.5 else 'flows'})")
        rad_s = args.drum * MAX_DRUM_SPEED
        print(f"  drum: {args.drum:+.2f} -> {rad_s:+.1f} rad/s -> "
              f"tip {abs(rad_s) * ROTOR_TIP_R:.2f} m/s "
              f"(at r = {ROTOR_TIP_R:.3f} m, the vane tip)")
        if abs(rad_s) > SPIN_LIMIT:
            print(f"  !! {abs(rad_s):.2f} rad/s is over the {SPIN_LIMIT:.2f} rad/s where "
                  "outward acceleration passes lunar gravity: regolith is thrown at the "
                  "shroud instead of carried inward")
        if args.drum * DIG_DRUM_SIGN < 0:
            print(f"  * this is the DUMP direction. Loading is {DIG_DRUM_SIGN:+.0f}; at "
                  f"{args.drum:+.2f} the channel runs outward and the drum empties. "
                  "Expect fill to fall, not rise.")

        boom, angle = boom_command_for_cut(u.cfg, args.cut)
        if args.boom is not None:
            boom = args.boom
            print(f"  boom command {boom:+.3f} (given directly, --cut ignored)\n")
        else:
            # --cut is measured to the SHROUD. The rotor sits inside it, so
            # the vane tips reach shallower than the number asked for, and
            # that is the number to watch: it decides how much of the rotor is
            # actually in soil.
            wall = SHROUD_OUT_R - ROTOR_TIP_R
            print(f"  to cut {args.cut:.3f} m -> arm {angle:+.3f} rad "
                  f"-> boom command {boom:+.3f}")
            print(f"  vane tips reach {args.cut - wall:.3f} m below grade")
            for line in cut_diagnosis(u.cfg, args.cut):
                print(f"  * {line}")
            print()

        obs, _ = env.reset()
        action = torch.zeros(u.num_envs, 4, device=u.device)
        # Throughput, because every training estimate for this task is
        # (env-steps needed) / (env-steps per second) and the second term is
        # the one nobody has. Wall clock over the whole run, so it includes
        # the MPM solve, the coupler and the reward -- which is what a trainer
        # actually pays for.
        t_start = time.perf_counter()

        peak = torch.zeros(u.num_envs, 2, device=u.device)
        peak_lip = torch.zeros(u.num_envs, 2, device=u.device)
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
            action[:, 2] = -1.0 + (boom + 1.0) * lower
            action[:, 3] = args.drum * spin

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
                      f"rew={float(rew[0]):+6.3f}")
            if bool(term.any()) or bool(trunc.any()):
                print(f"  t={t:5.1f}s  episode ended (terminated={bool(term.any())}, "
                      f"truncated={bool(trunc.any())}) -- bed and machine reset")

        wall = time.perf_counter() - t_start
        sps = steps * u.num_envs / max(wall, 1e-9)
        print("\n=== throughput ===")
        print(f"  {steps} steps x {u.num_envs} env in {wall:.1f} s "
              f"-> {sps:,.0f} env-steps/s at {u.cfg.bed_particles_per_env:,} particles")

        print("\n=== result ===")
        print(f"  peak fill per drum, per env (kg):\n{peak}")
        print(f"  peak in the shell wall band, never reaching the bore (kg):\n{peak_lip}")
        if float(peak.max()) < 1.0:
            voxel = u.cfg.voxel_size
            clear = 0.0819 - voxel
            print(f"\n  Fill never moved. The entry channel opens 0.0819 m and the coupler")
            print(f"  eats a whole voxel of it, leaving {clear:.4f} m clear -- "
                  f"{clear / voxel:.1f} particle")
            print("  spacings. Granular material ARCHES across an orifice narrower than")
            print("  about four to six grains: the grains jam against each other and the")
            print("  flow simply stops, however hard it is pushed. That is a property of")
            print("  the material, not of the drum, and no lip geometry defeats it.")
            print(f"\n  At this voxel the channel is {clear / voxel:.1f} grains wide. Options, cheapest first:")
            print(f"    env.voxel_size=0.02   -> {(0.0819 - 0.02) / 0.02:.1f} grains. More particles, but the")
            print("                             contact buffers are no longer the limit")
            print("    --cohesion 0           -> cohesive material arches MORE readily")
            print("    --drum +0.4            -> if fill RISES, DIG_DRUM_SIGN is inverted")
            print("                             and we have been running the dump direction")
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
            print("\n  If particles VISIBLY move but fill stays near zero, none of the")
            print("  above is wrong -- the drum is failing to carry soil. In order:")
            print("    --cut 0.18               get more of the bore under the surface;")
            print("                             at 0.12 only ~0.09 m of a 0.34 m bore is")
            print("                             below grade, so most of it never sees soil")
            print("    --cohesion 1500          cut material travels as a clod instead of")
            print("                             shearing off the lip and flowing back out.")
            print("                             NOT env.soil_cohesion=1500: Hydra applies")
            print("                             that after the material is already built")
            print("    --drum 0.25              slower; above the spin limit the vanes")
            print("                             throw regolith at the shroud instead of")
            print("                             carrying it inward")
            print(f"    --drum {-0.4 * DIG_DRUM_SIGN:+.1f}             runs the vanes backwards, which should")
            print("                             make fill FALL. If it rises instead,")
            print("                             RAKE_SIGN is mirrored -- flip it and")
            print("                             DIG_DRUM_SIGN follows")
        env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
