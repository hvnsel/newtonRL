# scripts/rft_pass.py
#
# Scripted pass of the front drum through the analytical (RFT) soil of
# mdp/rft.py, with no simulator. Over 2 s the drum lowers into a level bed and
# spins up; then it travels forward. --cut, --drum, --drive, --friction and
# --density mean what they do in dig_demo.py.
#
#   workstation/train.sh -p scripts/rft_pass.py
#   workstation/train.sh -p scripts/rft_pass.py --cut 0.05 --drive 0.1 --seconds 20

from __future__ import annotations

import argparse
import math

import torch

from luna_hifi_tasks.excavator.excavate.excavate_env_cfg import BORE_VOLUME
from luna_hifi_tasks.excavator.excavator import DIG_DRUM_SIGN, ROTOR_TIP_R, WHEEL_RADIUS
from luna_hifi_tasks.excavator.excavator_cfg import LUNAR_G, MAX_DRUM_SPEED, MAX_WHEEL_SPEED
from luna_hifi_tasks.excavator.mdp import rft

BED = (4.0, 1.6, 0.25)      # level bed: length along x, width, depth [m]
START_X = 1.0               # where the drum comes down [m]
CLEARANCE = 0.05            # vane tips above the bed before it does [m]
RAMP = 2.0                  # time to lower and spin up [s]
DT = 0.01                   # step [s]


def about_y(angle: float) -> torch.Tensor:
    """(1, 1, 4) quaternion (x, y, z, w) of a turn about +y."""
    return torch.tensor([[[0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)]]])


def main() -> None:
    p = argparse.ArgumentParser(description="Scripted drum pass through the RFT soil.")
    p.add_argument("--cut", type=float, default=0.073, help="vane tips below the bed surface, m")
    p.add_argument("--drum", type=float, default=1.0,
                   help="drum speed, a fraction of MAX_DRUM_SPEED in the digging direction")
    p.add_argument("--drive", type=float, default=0.05,
                   help="forward speed once the drum is down, a fraction of 1.5 m/s")
    p.add_argument("--tilt", type=float, default=0.096, help="shroud opening off straight down, rad")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--friction", type=float, default=0.84, help="soil friction, tan(phi)")
    p.add_argument("--density", type=float, default=1500.0, help="soil density, kg/m3")
    a = p.parse_args()

    soil = rft.Soil(density=a.density, friction=a.friction, gravity=LUNAR_G, lower=(0.0, -0.5 * BED[1]))
    nx, ny = round(BED[0] / soil.cell), round(BED[1] / soil.cell)
    height = torch.full((1, ny, nx), BED[2])
    fill = torch.zeros(1, 1)
    capacity = BORE_VOLUME * soil.density
    patches = rft.drum_patches()
    cells = rft.cell_centres(soil, nx, ny)

    speed = a.drive * MAX_WHEEL_SPEED * WHEEL_RADIUS
    full_spin = DIG_DRUM_SIGN * a.drum * MAX_DRUM_SPEED
    shroud_quat = about_y(a.tilt)
    axis = rft.drum_axis(shroud_quat)
    still = torch.zeros(1, 1, 3)
    angle = 0.0

    print(f"soil: density {soil.density:.0f} kg/m3, friction {soil.friction:.2f}, "
          f"gravity {soil.gravity:.2f} m/s2 -> stress scale {soil.zeta:.0f} N/m3")
    print(f"drum: cut {a.cut:.3f} m, {speed:.3f} m/s forward, {full_spin:+.2f} rad/s, holds {capacity:.1f} kg")
    print("force: soil on the drum. torque: soil on the rotor about +y, the axis it spins on.\n")
    print("   t [s]   x [m]  cut [m]  fill [kg]  force x [N]  force z [N]  torque [N m]")
    for i in range(round(a.seconds / DT)):
        t = i * DT
        ramp = min(t / RAMP, 1.0)
        cut = ramp * (a.cut + CLEARANCE) - CLEARANCE
        centre = torch.tensor([[[START_X + speed * max(t - RAMP, 0.0), 0.0, BED[2] + ROTOR_TIP_R - cut]]])
        lin = torch.tensor([[[speed, 0.0, 0.0] if t >= RAMP else [0.0, 0.0, -(a.cut + CLEARANCE) / RAMP]]])
        spin = full_spin * ramp
        angle += spin * DT

        shroud_force, _, rotor_force, rotor_torque = rft.drum_loads(
            patches, centre, lin, shroud_quat, still, about_y(angle), spin * axis, height, fill, capacity, soil
        )
        height, fill = rft.excavate(height, fill, cells, centre, axis, capacity, soil)

        if i % round(0.5 / DT) == 0:
            force = (shroud_force + rotor_force)[0, 0]
            print(f"  {t:6.1f}  {float(centre[0, 0, 0]):6.2f}  {cut:7.3f}  {float(fill):9.2f}  "
                  f"{float(force[0]):11.1f}  {float(force[2]):11.1f}  {float(rotor_torque[0, 0, 1]):12.2f}")

    removed = (BED[2] * nx * ny - float(height.sum())) * soil.cell ** 2 * soil.density
    print(f"\nsoil off the bed {removed:.2f} kg, in the drum {float(fill):.2f} kg")
    trench = BED[2] - height[0, ny // 2, ::5]
    print("cut depth along the bed's centreline, every 0.2 m [cm]:")
    print("  " + " ".join(f"{100.0 * float(d):4.1f}" for d in trench))


if __name__ == "__main__":
    main()
