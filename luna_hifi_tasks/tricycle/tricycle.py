# tricycle.py
# vibecoded tricycle asset for training
#
# The car, as an MJCF string. One steerable front wheel, two driven rear
# wheels, and a triangular frame connecting the three.
#
# This file is NOT read at training time. Training loads the converted USD in
# assets/tricycle/. The MJCF here is the *source* of that USD: whenever you
# change it, re-run the converter (see RUNBOOK.md, "Convert the tricycle
# asset") so the USD matches again.
#
# Body (plan view, +x forward):
#
#          rear_left                 apex / steering head
#             o----------------------o        <- left rail
#             |  [   deck   ]     .'          front wheel hangs below the apex
#             o----------------.'             <- right rail
#          rear_right
#           ^ rear crossbar
#
# Wheels are spheres, not cylinders: analytic spheres are the cheapest and most
# robust collider the MPM grid handles, and for "does the pipeline train" the
# contact patch shape is irrelevant. Wheel positions and radii are unchanged
# from the first working version, so nothing about the soil coupling moved.
#
# Every body with a joint has a geom with mass, so MuJoCo derives real inertia
# for all of them. A body with zero inertia produces NaNs in the solver.
#
# Frame: +x forward, +z up. Wheel bottoms sit at z = 0 in the chassis frame's
# rest pose (chassis centre at z = 0.14, wheel centres at z = 0.10, r = 0.10).

from __future__ import annotations

import math
import os
import tempfile

# ---------------------------------------------------------------------------
# Dimensions (metres, radians)
# ---------------------------------------------------------------------------

WHEEL_RADIUS = 0.10
CHASSIS_Z = 0.14          # chassis centre height with wheels resting on z = 0
WHEEL_Z = WHEEL_RADIUS - CHASSIS_Z   # wheel centre height relative to chassis (-0.04)

FRONT_X = 0.22            # steering axis / front wheel centre
REAR_X = -0.15            # rear axle
REAR_Y = 0.16             # rear wheel centre, either side

# Triangle: apex at the steering head, base between the rear wheels.
APEX = (FRONT_X, 0.0)
REAR_CORNER_Y = 0.13      # inside the rear wheels, so the rails end inside them
RAIL_HALF_W = 0.02        # rail cross-section is 4 cm wide x 3 cm tall
RAIL_HALF_H = 0.015

# Masses: same 5.0 kg chassis total as the original box body.
RAIL_MASS = 0.8
CROSSBAR_MASS = 0.8
DECK_MASS = 2.6

JOINT_STEER = "steer"
JOINT_FRONT = "front_wheel"
JOINT_REAR = ["rear_left", "rear_right"]

FRAME_RGBA = "0.2 0.4 0.8 1"
DECK_RGBA = "0.55 0.6 0.7 1"
WHEEL_RGBA = "0.15 0.15 0.15 1"

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rail(name: str, start: tuple[float, float], end: tuple[float, float], mass: float) -> str:
    """A box geom whose local x axis runs from `start` to `end` (chassis frame, z = 0)."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    half_len = 0.5 * math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    cx, cy = 0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1])
    return (
        f'<geom name="{name}" type="box" pos="{cx:.4f} {cy:.4f} 0" euler="0 0 {yaw:.6f}" '
        f'size="{half_len:.4f} {RAIL_HALF_W} {RAIL_HALF_H}" mass="{mass}" rgba="{FRAME_RGBA}"/>'
    )


LEFT_RAIL = _rail("left_rail", (REAR_X, REAR_CORNER_Y), APEX, RAIL_MASS)
RIGHT_RAIL = _rail("right_rail", (REAR_X, -REAR_CORNER_Y), APEX, RAIL_MASS)

# Base of the triangle. Runs slightly past the rail ends so the corners are closed.
CROSSBAR = (
    f'<geom name="crossbar" type="box" pos="{REAR_X} 0 0" '
    f'size="{RAIL_HALF_W} {REAR_CORNER_Y + RAIL_HALF_W} {RAIL_HALF_H}" '
    f'mass="{CROSSBAR_MASS}" rgba="{FRAME_RGBA}"/>'
)

# Flat floor pan inside the rear half of the triangle. Carries most of the mass.
DECK = (
    f'<geom name="deck" type="box" pos="-0.08 0 0" size="0.06 0.065 0.01" '
    f'mass="{DECK_MASS}" rgba="{DECK_RGBA}"/>'
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

TRICYCLE_MJCF = f"""
<mujoco model="tricycle">
  <compiler angle="radian" autolimits="true"/>
  <default>
    <geom condim="3" friction="0.8 0.005 0.0001"/>
    <joint damping="0.02"/>
  </default>

  <worldbody>
    <body name="chassis" pos="0 0 {CHASSIS_Z}">
      <freejoint name="root"/>
      {LEFT_RAIL}
      {RIGHT_RAIL}
      {CROSSBAR}
      {DECK}

      <!-- steering fork: hinge about z, carries the free-rolling front wheel -->
      <body name="fork" pos="{FRONT_X} 0 {WHEEL_Z}">
        <joint name="{JOINT_STEER}" type="hinge" axis="0 0 1" range="-0.6 0.6"/>
        <geom type="capsule" fromto="0 0 0.02 0 0 -0.02" size="0.012" mass="0.1"/>
        <body name="front_wheel_body" pos="0 0 0">
          <joint name="{JOINT_FRONT}" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="{WHEEL_RADIUS}" mass="0.5" rgba="{WHEEL_RGBA}"/>
        </body>
      </body>

      <!-- driven rear wheels -->
      <body name="rear_left_body" pos="{REAR_X} {REAR_Y} {WHEEL_Z}">
        <joint name="{JOINT_REAR[0]}" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="{WHEEL_RADIUS}" mass="0.5" rgba="{WHEEL_RGBA}"/>
      </body>
      <body name="rear_right_body" pos="{REAR_X} {-REAR_Y} {WHEEL_Z}">
        <joint name="{JOINT_REAR[1]}" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="{WHEEL_RADIUS}" mass="0.5" rgba="{WHEEL_RGBA}"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def write_mjcf(directory: str | None = None) -> str:
    """Write the MJCF to disk and return its path. Idempotent.

    Used only by the offline USD conversion step; nothing calls it at training time.
    """
    directory = directory or os.path.join(tempfile.gettempdir(), "luna_hifi_assets")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "tricycle.xml")
    with open(path, "w") as fh:
        fh.write(TRICYCLE_MJCF.strip() + "\n")
    return path
