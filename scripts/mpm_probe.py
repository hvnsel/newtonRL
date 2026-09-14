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
LADDER = [
    # label,                voxel, len,  wid, depth, cap, couple
    #
    # The first four are the configurations that settled it: survival tracked
    # the sparse-grid cap exactly, at every voxel and with the collider set
    # varied, and nothing else moved the boundary. Keep them as a regression --
    # if "was 6882, DIED" fails again, the leaf-node ratio has regressed.
    ("baseline",             0.05, 0.90, 1.10, 0.16,  8.0, "all"),
    ("was 3168, OK",         0.05, 1.80, 1.10, 0.16,  8.0, "front"),
    ("was 5544, DIED",       0.05, 1.80, 1.10, 0.32,  8.0, "front"),
    ("was 6882, DIED",       0.03, 0.90, 1.10, 0.16,  8.0, "front"),
    # Now the point of the exercise: a bed worth digging, at a voxel the drum
    # can actually pass soil through.
    ("0.03, all coupled",    0.03, 0.90, 1.10, 0.16,  8.0, "all"),
    ("0.03, 10k",            0.03, 1.00, 1.10, 0.24,  8.0, "all"),
    ("0.03, 20k",            0.03, 1.60, 1.30, 0.30,  8.0, "all"),
    ("0.03, 40k",            0.03, 2.60, 1.60, 0.36,  8.0, "all"),
    ("0.025, 40k",          0.025, 1.80, 1.30, 0.30,  8.0, "all"),
    ("0.03, 80k",            0.03, 4.00, 1.80, 0.42,  8.0, "all"),
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
    print(f"{'rung':<24} {'voxel':>6} {'bed':>18} {'cap':>5} {'couple':>7}  result")
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
        print(f"{label:<24} {voxel:6.3f} {bed:>18} {cap:5.0f} {couple:>7}  "
              f"{'OK' + n if ok else 'DIED (rc=%s)' % r.returncode}")
        if ok:
            last_ok = label
        sys.stdout.flush()
    print(f"\nlast rung that survived: {last_ok}")
    print("The first DIED tells you which knob is the real limit. Paste the table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
