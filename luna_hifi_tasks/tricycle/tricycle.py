# tricycle.py
#
# The car, as an MJCF string. One steerable front wheel, two driven rear
# wheels. Written to disk at import so Isaac Lab's MjcfFileCfg can point at it.
#
# Wheels are spheres, not cylinders: analytic spheres are the cheapest and most
# robust collider the MPM grid handles, and for "does the pipeline train" the
# contact patch shape is irrelevant.
#
# Every body with a joint has a geom with mass, so MuJoCo derives real inertia
# for all of them. That is the bug that made the standalone spheres NaN --
# there is no zero-inertia body here.
#
# Frame: +x forward, +z up. Wheel bottoms sit at z = 0 in the chassis frame's
# rest pose (chassis centre at z = 0.14, wheel centres at z = 0.10, r = 0.10).
#
# Standalone use (viewer / sanity check, not training):
#     builder.add_mjcf(TRICYCLE_MJCF)

from __future__ import annotations

import os
import tempfile

WHEEL_RADIUS = 0.10
CHASSIS_Z = 0.14          # chassis centre height with wheels resting on z = 0

JOINT_STEER = "steer"
JOINT_FRONT = "front_wheel"
JOINT_REAR = ["rear_left", "rear_right"]

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
      <geom name="chassis_geom" type="box" size="0.20 0.12 0.04" mass="5.0" rgba="0.2 0.4 0.8 1"/>

      <!-- steering fork: hinge about z, carries the free-rolling front wheel -->
      <body name="fork" pos="0.22 0 -0.04">
        <joint name="{JOINT_STEER}" type="hinge" axis="0 0 1" range="-0.6 0.6"/>
        <geom type="capsule" fromto="0 0 0.02 0 0 -0.02" size="0.012" mass="0.1"/>
        <body name="front_wheel_body" pos="0 0 0">
          <joint name="{JOINT_FRONT}" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="{WHEEL_RADIUS}" mass="0.5" rgba="0.15 0.15 0.15 1"/>
        </body>
      </body>

      <!-- driven rear wheels -->
      <body name="rear_left_body" pos="-0.15 0.16 -0.04">
        <joint name="{JOINT_REAR[0]}" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="{WHEEL_RADIUS}" mass="0.5" rgba="0.15 0.15 0.15 1"/>
      </body>
      <body name="rear_right_body" pos="-0.15 -0.16 -0.04">
        <joint name="{JOINT_REAR[1]}" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="{WHEEL_RADIUS}" mass="0.5" rgba="0.15 0.15 0.15 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def write_mjcf(directory: str | None = None) -> str:
    """Write the MJCF to disk and return its path. Idempotent."""
    directory = directory or os.path.join(tempfile.gettempdir(), "luna_hifi_assets")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "tricycle.xml")
    with open(path, "w") as fh:
        fh.write(TRICYCLE_MJCF.strip() + "\n")
    return path


TRICYCLE_MJCF_PATH = write_mjcf()
