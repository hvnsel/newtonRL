# scripts/mpm_probe.py
#
# Find what this GPU will actually take, by trying it.
#
#   isaaclab -p scripts/mpm_probe.py                 # walk the ladder
#   isaaclab -p scripts/mpm_probe.py --child ...     # one trial (used internally)
#
# Why a separate script. An MPM config that exceeds some internal capacity does
# not raise -- it corrupts the CUDA context and takes the process with it, as a
# CUDA 700 storm or a 0xC0000374 on Windows. You cannot catch that and carry on,
# so each trial has to be its own process, and the only honest way to find the
# limit is to walk one knob at a time and see where it dies.
#
# This exists because three separate theories about that limit -- particle
# count, sparse-grid capacity, collider footprint -- were each argued from
# first principles and each turned out to be wrong or incomplete. The ladder
# below changes ONE thing per rung against a known-good baseline, so whatever
# it finds is a measurement rather than an argument.

from __future__ import annotations

import argparse
import math
import subprocess
import sys


def child(args) -> int:
    import gymnasium as gym
    import torch

    from isaaclab.app import add_launcher_args, launch_simulation
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import setup_preset_cli
    from isaaclab_tasks.utils.hydra import resolve_task_config
    import luna_hifi_tasks  # noqa: F401
    from luna_hifi_tasks.excavator.excavator_cfg import (
        FRONT_DRUM_ONLY_REGEX,
        SOIL_CONTACT_BODIES_REGEX,
    )

    p = argparse.ArgumentParser()
    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=[])
    launch, hydra_args = setup_preset_cli(p, [])
    sys.argv = [sys.argv[0]] + hydra_args

    cfg, _ = resolve_task_config(args.task, "")
    cfg.scene.num_envs = 1
    cfg.voxel_size = args.voxel
    cfg.bed_x = (1.05, 1.05 + args.bed_len)
    cfg.bed_y = (-0.5 * args.bed_wid, 0.5 * args.bed_wid)
    cfg.bed_depth = args.bed_depth
    cfg.grid_cap_multiplier = args.cap_mult
    cfg.soil_contact_regex = (
        FRONT_DRUM_ONLY_REGEX if args.couple == "front" else SOIL_CONTACT_BODIES_REGEX
    )
    cfg.__post_init__()
    cfg.apply_soil_material()

    launch.device = cfg.sim.device
    cfg.validate()
    with launch_simulation(cfg, launch):
        env = gym.make(args.task, cfg=cfg)
        u = env.unwrapped
        action = torch.zeros(u.num_envs, 4, device=u.device)
        action[:, 2] = 0.3          # boom down a little, so the drum meets soil
        action[:, 3] = 0.4
        env.reset()
        for _ in range(args.steps):
            env.step(action)
        env.close()
    print(f"PROBE_OK particles={cfg.bed_particles_per_env}")
    return 0


# One knob changed per rung, against a baseline that is known to run.
# Hold ONE of (particles, grid cap) fixed while the other moves. The previous
# ladder could not tell them apart: active is next_pow2(cap * particles), so
# every rung moved both together and "died above 65,536 cells" and "died above
# ~4,100 particles" were the same sentence. These rungs break that.
#
# All at a 0.05 m voxel and the same bed footprint, so the only things changing
# are the two under test.
LADDER = [
    # label,                voxel, len,  wid, depth, cap, couple
    ("control 3168p cap8",   0.05, 1.80, 1.10, 0.16,  8.0, "all"),
    # Same particles, 4x the grid. If THIS dies, the grid is the constraint.
    ("3168p cap32",          0.05, 1.80, 1.10, 0.16, 32.0, "all"),
    # Same particles, quarter the grid. If the control lives and this dies too,
    # the tree floors are being starved rather than the grid overflowing.
    ("3168p cap2",           0.05, 1.80, 1.10, 0.16,  2.0, "all"),
    # Particles that DIED at cap 8, now with a quarter of the grid. If this
    # lives, it was the grid all along and the fix is the cap, not the bed.
    ("5544p cap2",           0.05, 1.80, 1.10, 0.32,  2.0, "all"),
    # And bisect the particle boundary itself, grid held as low as it goes.
    ("3960p cap2",           0.05, 1.80, 1.10, 0.25,  2.0, "all"),
    ("4752p cap2",           0.05, 1.80, 1.10, 0.30,  2.0, "all"),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Find the MPM limit on this GPU by trying it.")
    p.add_argument("--task", default="Luna-Excavator-Excavate-Micro")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--child", action="store_true", help="run ONE trial (internal)")
    p.add_argument("--voxel", type=float, default=0.05)
    p.add_argument("--bed_len", type=float, default=0.90)
    p.add_argument("--bed_wid", type=float, default=1.10)
    p.add_argument("--bed_depth", type=float, default=0.16)
    p.add_argument("--cap_mult", type=float, default=8.0)
    p.add_argument("--couple", choices=("all", "front"), default="all")
    args, _ = p.parse_known_args()

    if args.child:
        return child(args)

    print("Each rung is its own process: an over-capacity MPM config kills the")
    print("process rather than raising, so it cannot be caught and retried.\n")
    print(f"{'rung':<24} {'voxel':>6} {'bed':>18} {'cap':>5} {'couple':>7} "
          f"{'parts':>8} {'cells':>10}  result")
    last_ok = None
    for label, voxel, blen, bwid, bdep, cap, couple in LADDER:
        cmd = [sys.executable, __file__, "--child", "--task", args.task,
               "--steps", str(args.steps), "--voxel", str(voxel),
               "--bed_len", str(blen), "--bed_wid", str(bwid),
               "--bed_depth", str(bdep), "--cap_mult", str(cap), "--couple", couple]
        r = subprocess.run(cmd, capture_output=True, text=True)
        ok = "PROBE_OK" in r.stdout
        n = ""
        for line in r.stdout.splitlines():
            if line.startswith("PROBE_OK"):
                n = " " + line.split("particles=")[1] + "p"
        bed = f"{blen:.2f}x{bwid:.2f}x{bdep:.2f}"
        # Work out the count here too, so a rung that dies still says how big
        # it was: the child takes the process with it and prints nothing.
        want = 1
        for extent in (blen, bwid, bdep):
            want *= max(math.ceil(extent / voxel), 1)
        cells = 1 << max(int(int(cap * want) - 1).bit_length(), 1)
        print(f"{label:<24} {voxel:6.3f} {bed:>18} {cap:5.0f} {couple:>7} "
              f"{want:7,d}p {cells:9,d}c  "
              f"{'OK' + n if ok else 'DIED (rc=%s)' % r.returncode}")
        if ok:
            last_ok = label
        sys.stdout.flush()
    print(f"\nlast rung that survived: {last_ok}")
    print("\nRead it like this:")
    print("  3168p cap32 dies        -> the GRID is the constraint, shrink the cap")
    print("  5544p cap2 lives        -> same conclusion, and the bed was never the problem")
    print("  only the particle count -> it is the particles, and the cap is a red herring")
    print("     tracks the deaths       (which is what the last ladder could not tell us)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
