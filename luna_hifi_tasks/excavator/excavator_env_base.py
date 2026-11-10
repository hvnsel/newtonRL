# What navigate and excavate share: joint and body handles, skid-steer drive,
# joint targets, proprioception, failure terminations and robot reset.

from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .excavator import (
    JOINT_ARMS,
    JOINT_DRUMS,
    JOINT_SHROUDS,
    JOINT_WHEELS,
    JOINT_WHEELS_LEFT,
    JOINT_WHEELS_RIGHT,
    BODY_DRUMS,
    BODY_SHROUDS,
    ROTOR_VANES,
    TRACK,
    WHEEL_RADIUS,
)
from .excavator_cfg import (
    ARM_EFFORT,
    DRUM_EFFORT,
    MAX_DRUM_SPEED,
    MAX_WHEEL_SPEED,
    MAX_YAW_SPEED,
)
from .mdp import rewards as R
from .mdp.terrain import yaw_from_quat


class ExcavatorEnvBase(DirectRLEnv):
    """Subclasses own actions, observations and rewards."""

    cfg: object

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._wheel_ids, _ = self.robot.find_joints(JOINT_WHEELS, preserve_order=True)
        self._left_ids, _ = self.robot.find_joints(JOINT_WHEELS_LEFT, preserve_order=True)
        self._right_ids, _ = self.robot.find_joints(JOINT_WHEELS_RIGHT, preserve_order=True)
        self._arm_ids, _ = self.robot.find_joints(JOINT_ARMS, preserve_order=True)
        self._drum_ids, _ = self.robot.find_joints(JOINT_DRUMS, preserve_order=True)
        self._shroud_ids, _ = self.robot.find_joints(JOINT_SHROUDS, preserve_order=True)
        self._drum_body_ids, _ = self.robot.find_bodies(BODY_DRUMS, preserve_order=True)
        self._shroud_body_ids, _ = self.robot.find_bodies(BODY_SHROUDS, preserve_order=True)
        # Last commanded body speeds.
        self._forward_cmd = torch.zeros(self.num_envs, device=self.device)   # m/s
        self._yaw_cmd = torch.zeros(self.num_envs, device=self.device)       # rad/s

    def _setup_scene(self):
        self.robot: Articulation = self.scene["excavator"]

    # --- joint targets ---

    def _apply_drive(self, forward: torch.Tensor, yaw: torch.Tensor) -> None:
        """Wheel velocity targets for skid steer from forward and yaw in [-1, 1]."""
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
        """Position target (E,) rad for both arms."""
        self.robot.set_joint_position_target(angle.unsqueeze(-1).expand(-1, 2), joint_ids=self._arm_ids)

    def _apply_drums(self, speed: torch.Tensor) -> None:
        """Velocity target (E,) rad/s for both drums."""
        self.robot.set_joint_velocity_target(speed.unsqueeze(-1).expand(-1, 2), joint_ids=self._drum_ids)

    def _apply_shrouds(self, angle: torch.Tensor) -> None:
        """Position target (E,) rad for both shrouds (the drum inlets)."""
        self.robot.set_joint_position_target(angle.unsqueeze(-1).expand(-1, 2), joint_ids=self._shroud_ids)

    # --- observations ---

    def _proprio(self) -> dict[str, torch.Tensor]:
        d = self.robot.data
        tau = d.applied_torque.torch
        # Rotor phase within a vane pocket.
        rotor = ROTOR_VANES * d.joint_pos.torch[:, self._drum_ids]
        return {
            "base_lin_vel": d.root_lin_vel_b.torch,
            "base_ang_vel": d.root_ang_vel_b.torch,
            "projected_gravity": d.projected_gravity_b.torch,
            "wheel_vel": d.joint_vel.torch[:, self._wheel_ids] / MAX_WHEEL_SPEED,
            "arm_pos": d.joint_pos.torch[:, self._arm_ids],
            "arm_vel": d.joint_vel.torch[:, self._arm_ids],
            "drum_vel": d.joint_vel.torch[:, self._drum_ids] / MAX_DRUM_SPEED,
            "drum_phase": torch.cat([torch.sin(rotor), torch.cos(rotor)], dim=-1),
            "shroud_pos": d.joint_pos.torch[:, self._shroud_ids],
            "arm_torque": tau[:, self._arm_ids] / ARM_EFFORT,
            "drum_torque": tau[:, self._drum_ids] / DRUM_EFFORT,
        }

    def _base_yaw(self) -> torch.Tensor:
        return yaw_from_quat(self.robot.data.root_quat_w.torch)

    def _drum_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(E, 2, 3) positions and (E, 2, 4) xyzw quaternions of the drum bodies."""
        d = self.robot.data
        return d.body_pos_w.torch[:, self._drum_body_ids], d.body_quat_w.torch[:, self._drum_body_ids]

    def _shroud_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(E, 2, 3) positions and (E, 2, 4) xyzw quaternions of the shroud bodies,
        on the drum axis; the inlet is centred on each shroud's local -z."""
        d = self.robot.data
        return d.body_pos_w.torch[:, self._shroud_body_ids], d.body_quat_w.torch[:, self._shroud_body_ids]

    # --- terminations and reset ---

    def _failed(self) -> torch.Tensor:
        """Non-finite state, beyond max_env_excursion, over 20 m/s, or tipped."""
        d = self.robot.data
        pos = d.root_pos_w.torch - self.scene.env_origins
        bad = R.nonfinite_state(pos, d.root_lin_vel_b.torch, d.joint_vel.torch)
        bad |= pos.abs().max(dim=-1).values > self.cfg.max_env_excursion
        bad |= d.root_lin_vel_b.torch.abs().max(dim=-1).values > 20.0
        bad |= R.tipped(d.projected_gravity_b.torch, math.cos(self.cfg.max_tilt_rad))
        return bad

    def _exit_stats(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Reset count and the share terminated and timed out."""
        return {
            "Episode/resets": float(env_ids.numel()),
            "Episode/terminated": float(self.reset_terminated[env_ids].float().mean()),
            "Episode/timeout": float(self.reset_time_outs[env_ids].float().mean()),
        }

    def _reset_robot(
        self,
        env_ids: torch.Tensor,
        yaw: torch.Tensor,
        arm_angle: float | torch.Tensor,
        xy_offset: torch.Tensor | None = None,
        drum_angle: torch.Tensor | None = None,
        z_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Place the machine at its default root state plus the env origin,
        xy_offset and z_offset, facing yaw, arms at arm_angle (and drums at
        drum_angle), at rest. Returns the world xy written; robot.data still
        holds the old pose until the next sim step."""
        root = self.robot.data.default_root_state.torch[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        if xy_offset is not None:
            root[:, :2] += xy_offset
        if z_offset is not None:
            root[:, 2] += z_offset
        root[:, 3] = 0.0                                      # quaternion (x, y, z, w), yaw only
        root[:, 4] = 0.0
        root[:, 5] = torch.sin(0.5 * yaw)
        root[:, 6] = torch.cos(0.5 * yaw)
        root[:, 7:] = 0.0
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)

        jpos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        jvel = torch.zeros_like(jpos)
        jpos[:, self._arm_ids] = torch.as_tensor(arm_angle, device=self.device).reshape(-1, 1)
        if drum_angle is not None:
            jpos[:, self._drum_ids] = drum_angle.unsqueeze(-1)
        self.robot.write_joint_state_to_sim(jpos, jvel, None, env_ids)

        self._forward_cmd[env_ids] = 0.0
        self._yaw_cmd[env_ids] = 0.0
        return root[:, :2].clone()
