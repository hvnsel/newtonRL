# excavator_env_base.py
#
# What the navigate and excavate environments share: the machine, its joint
# and body handles, the skid-steer drive mapping, proprioceptive observation
# terms, the failure terminations, and root reset.
#
# Assets are fetched from the scene in _setup_scene, never constructed there;
# joint ids resolve once in __init__; resets write root pose, root velocity and
# joint state separately.
#
# write_root_pose_to_sim, set_joint_velocity_target and friends are marked
# deprecated in favour of the *_index variants and the actuator command API,
# but still carry the same positional signatures.

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .excavator import (
    JOINT_ARMS,
    JOINT_DRUMS,
    JOINT_WHEELS,
    JOINT_WHEELS_LEFT,
    JOINT_WHEELS_RIGHT,
    BODY_DRUMS,
    TRACK,
    WHEEL_RADIUS,
)
from .excavator_cfg import MAX_DRUM_SPEED, MAX_WHEEL_SPEED, MAX_YAW_SPEED
from .mdp import rewards as R
from .mdp.terrain import yaw_from_quat


class ExcavatorEnvBase(DirectRLEnv):
    """Shared machinery. Subclasses own actions, observations and rewards."""

    # subclasses set these
    cfg: object

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._wheel_ids, _ = self.robot.find_joints(JOINT_WHEELS, preserve_order=True)
        self._left_ids, _ = self.robot.find_joints(JOINT_WHEELS_LEFT, preserve_order=True)
        self._right_ids, _ = self.robot.find_joints(JOINT_WHEELS_RIGHT, preserve_order=True)
        self._arm_ids, _ = self.robot.find_joints(JOINT_ARMS, preserve_order=True)
        self._drum_ids, _ = self.robot.find_joints(JOINT_DRUMS, preserve_order=True)
        self._drum_body_ids, _ = self.robot.find_bodies(BODY_DRUMS, preserve_order=True)
        assert len(self._wheel_ids) == 4 and len(self._arm_ids) == 2 and len(self._drum_ids) == 2, (
            "joint lookup failed -- the USD's joint names do not match excavator.py; "
            f"have {self.robot.joint_names}"
        )
        assert len(self._drum_body_ids) == 2, (
            f"drum body lookup failed; have {self.robot.body_names}"
        )

        # commanded speeds, kept so rewards can compare asked vs achieved
        self._forward_cmd = torch.zeros(self.num_envs, device=self.device)   # m/s
        self._yaw_cmd = torch.zeros(self.num_envs, device=self.device)       # rad/s

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # InteractiveScene already spawned everything in the scene cfg.
        self.robot: Articulation = self.scene["excavator"]

    # ------------------------------------------------------------------
    # drive
    # ------------------------------------------------------------------

    def _apply_drive(self, forward: torch.Tensor, yaw: torch.Tensor) -> None:
        """Skid steer from [forward, yaw] in [-1, 1].

        Commanded in (forward, yaw) rather than (left, right) on purpose:
        driving straight is then yaw = 0, which is the mean of the untrained
        policy, instead of two Gaussian samples having to match exactly.
        """
        v = forward * MAX_WHEEL_SPEED * WHEEL_RADIUS          # m/s
        w = yaw * MAX_YAW_SPEED                               # rad/s
        half_track = 0.5 * TRACK
        v_left = (v - w * half_track) / WHEEL_RADIUS          # rad/s
        v_right = (v + w * half_track) / WHEEL_RADIUS
        self.robot.set_joint_velocity_target(v_left.unsqueeze(-1).expand(-1, 2), joint_ids=self._left_ids)
        self.robot.set_joint_velocity_target(v_right.unsqueeze(-1).expand(-1, 2), joint_ids=self._right_ids)
        self._forward_cmd[:] = v
        self._yaw_cmd[:] = w

    def _apply_arms(self, angle: torch.Tensor) -> None:
        """Position target for BOTH arms. (E,) rad, same value each side."""
        self.robot.set_joint_position_target(angle.unsqueeze(-1).expand(-1, 2), joint_ids=self._arm_ids)

    def _apply_drums(self, speed: torch.Tensor) -> None:
        """Velocity target for BOTH drums, (E,) rad/s. Same sign on each is the
        counter-rotating, reaction-cancelling dig -- see excavator.py."""
        self.robot.set_joint_velocity_target(speed.unsqueeze(-1).expand(-1, 2), joint_ids=self._drum_ids)

    # ------------------------------------------------------------------
    # observation pieces
    # ------------------------------------------------------------------

    def _proprio(self) -> dict[str, torch.Tensor]:
        d = self.robot.data
        return {
            "base_lin_vel": d.root_lin_vel_b.torch,
            "base_ang_vel": d.root_ang_vel_b.torch,
            "projected_gravity": d.projected_gravity_b.torch,
            "wheel_vel": d.joint_vel.torch[:, self._wheel_ids] / MAX_WHEEL_SPEED,
            "arm_pos": d.joint_pos.torch[:, self._arm_ids],
            "arm_vel": d.joint_vel.torch[:, self._arm_ids],
            "drum_vel": d.joint_vel.torch[:, self._drum_ids] / MAX_DRUM_SPEED,
        }

    def _base_yaw(self) -> torch.Tensor:
        return yaw_from_quat(self.robot.data.root_quat_w.torch)

    def _drum_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(E, 2, 3) positions and (E, 2, 4) xyzw quaternions of the drum bodies."""
        d = self.robot.data
        return d.body_pos_w.torch[:, self._drum_body_ids], d.body_quat_w.torch[:, self._drum_body_ids]

    # ------------------------------------------------------------------
    # failure terminations shared by both tasks
    # ------------------------------------------------------------------

    def _failed(self) -> torch.Tensor:
        d = self.robot.data
        pos = d.root_pos_w.torch - self.scene.env_origins
        bad = R.nonfinite_state(pos, d.root_lin_vel_b.torch, d.joint_vel.torch)
        bad |= pos.abs().max(dim=-1).values > self.cfg.max_env_excursion
        bad |= d.root_lin_vel_b.torch.abs().max(dim=-1).values > 20.0
        bad |= R.tipped(d.projected_gravity_b.torch, math.cos(self.cfg.max_tilt_rad))
        return bad

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def _reset_robot(
        self,
        env_ids: torch.Tensor,
        yaw: torch.Tensor,
        arm_angle: float,
        xy_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Put the machine at its env origin (+ `xy_offset`) facing `yaw`, arms
        at `arm_angle`, everything else at rest. Returns the world xy written.

        Callers must use the RETURNED position for anything else they do in
        the same reset (goal sampling, initial distances), not re-read
        data.root_pos_w: the data buffers are timestamped against the sim and
        do not refresh until the next step, so a read right after a write
        returns the pose from before the reset. The tricycle env keeps to the
        same discipline (it uses its written root_state, never a re-read).
        """
        root = self.robot.data.default_root_state.torch[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        if xy_offset is not None:
            root[:, :2] += xy_offset
        # yaw-only orientation, (x, y, z, w)
        root[:, 3] = 0.0
        root[:, 4] = 0.0
        root[:, 5] = torch.sin(0.5 * yaw)
        root[:, 6] = torch.cos(0.5 * yaw)
        root[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)

        jpos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        jvel = torch.zeros_like(jpos)
        jpos[:, self._arm_ids] = arm_angle
        self.robot.write_joint_state_to_sim(jpos, jvel, None, env_ids)

        self._forward_cmd[env_ids] = 0.0
        self._yaw_cmd[env_ids] = 0.0
        return root[:, :2].clone()
