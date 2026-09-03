# excavation_env.py
#
# DirectRLEnv for rover excavation on Newton implicit MPM soil.
#
# What Isaac Lab does for you here, that newton_soil.py used to do:
#   - registers mpm:* custom attributes on the builder
#   - builds SolverImplicitMPM + MJWarp and couples them
#   - steps both, calls project_outside, harvests soil impulses into body_f
#   - clones the scene across num_envs and gives you batched tensors
#
# What is yours: actions -> joint targets, particles -> observations, reward,
# termination, reset. All four are stubbed below at the level of "shapes are
# right, numbers are placeholders".
#
# STUB / VERIFY markers as in excavation_env_cfg.py.

from __future__ import annotations

import torch
import warp as wp

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms

# VERIFY: import path and the data attributes used in _particle_state().
from isaaclab_newton.assets import MPMObject

from .excavation_env_cfg import RoverExcavationEnvCfg
from .soil import bucket_fill_kernel, height_map_kernel


class RoverExcavationEnv(DirectRLEnv):
    cfg: RoverExcavationEnvCfg

    def __init__(self, cfg: RoverExcavationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -- joint / body indices
        self._wheel_ids, _ = self.rover.find_joints(self.cfg.wheel_joint_names)
        self._arm_ids, _ = self.rover.find_joints(self.cfg.arm_joint_names)
        self._bucket_id, _ = self.rover.find_bodies(self.cfg.bucket_body_name)
        self._bucket_id = self._bucket_id[0]

        n_wheels, n_arm = len(self._wheel_ids), len(self._arm_ids)
        assert n_wheels == self.cfg.num_wheels and n_arm == self.cfg.num_arm_joints, (
            f"joint regexes matched {n_wheels} wheels / {n_arm} arm joints; "
            f"cfg says {self.cfg.num_wheels} / {self.cfg.num_arm_joints}"
        )

        # -- action buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._arm_targets = self.rover.data.default_joint_pos[:, self._arm_ids].clone()
        self._arm_lower = self.rover.data.soft_joint_pos_limits[:, self._arm_ids, 0]
        self._arm_upper = self.rover.data.soft_joint_pos_limits[:, self._arm_ids, 1]

        # -- observation buffers (torch, fed to the policy)
        hm_cells = self.cfg.height_map_rows * self.cfg.height_map_cols
        self._height_map = torch.zeros(self.num_envs, hm_cells, device=self.device)
        self._bucket_fill = torch.zeros(self.num_envs, device=self.device)

        # -- warp views over the same memory for the kernels
        self._height_map_wp = wp.from_torch(self._height_map, dtype=float)
        self._bucket_fill_wp = wp.from_torch(self._bucket_fill, dtype=float)
        self._bucket_xform_wp = wp.zeros(self.num_envs, dtype=wp.transform, device=self.device)

        # -- reward bookkeeping
        self._prev_fill = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ scene

    def _setup_scene(self):
        self.rover = Articulation(self.cfg.scene.rover)
        self.soil = MPMObject(self.cfg.scene.soil)

        self.cfg.scene.ground.func("/World/ground", self.cfg.scene.ground)
        self.cfg.scene.light.func("/World/Light", self.cfg.scene.light)

        # Newton handles physics replication itself; Isaac Lab's clone step is
        # still what assigns env origins.
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["rover"] = self.rover
        self.scene.extras["soil"] = self.soil     # VERIFY: which scene bucket MPMObject registers under

    # ---------------------------------------------------------------- actions

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions[:] = self._actions
        self._actions[:] = actions.clamp(-1.0, 1.0)

        # arm: integrate deltas into a persistent target, clamp to soft limits
        arm_delta = self._actions[:, self.cfg.num_wheels:] * self.cfg.arm_action_scale
        self._arm_targets = torch.clamp(self._arm_targets + arm_delta, self._arm_lower, self._arm_upper)

    def _apply_action(self):
        wheel_vel = self._actions[:, : self.cfg.num_wheels] * self.cfg.max_wheel_speed
        self.rover.set_joint_velocity_target(wheel_vel, joint_ids=self._wheel_ids)
        self.rover.set_joint_position_target(self._arm_targets, joint_ids=self._arm_ids)

    # ----------------------------------------------------------- observations

    def _particle_state(self):
        """(pos, mass) as warp (num_envs, n_per_env) arrays in env-local frame.

        VERIFY the attribute names on MPMObject.data. Whatever they are, the
        contract the kernels need is: positions contiguous per env, world
        origin already subtracted.
        """
        pos_w = self.soil.data.particle_pos_w                      # (num_envs, N, 3) torch
        pos_local = pos_w - self.scene.env_origins.unsqueeze(1)
        mass = self.soil.data.particle_mass                        # (num_envs, N) torch
        return wp.from_torch(pos_local, dtype=wp.vec3), wp.from_torch(mass, dtype=float)

    def _update_soil_observations(self):
        pos_wp, mass_wp = self._particle_state()
        num_envs, n_per_env = pos_wp.shape

        # height-map: reset to floor, then max-scatter
        self._height_map.fill_(0.0)
        wp.launch(
            height_map_kernel,
            dim=(num_envs, n_per_env),
            inputs=[
                pos_wp,
                wp.vec2(*self.cfg.height_map_origin),
                float(self.cfg.height_map_cell),
                int(self.cfg.height_map_rows),
                int(self.cfg.height_map_cols),
                0.0,
                self._height_map_wp,
            ],
            device=self.device,
        )

        # bucket fill: transform particles into the bucket frame and box-test
        bucket_pos_w = self.rover.data.body_pos_w[:, self._bucket_id] - self.scene.env_origins
        bucket_quat_w = self.rover.data.body_quat_w[:, self._bucket_id]        # (w, x, y, z)
        xf = torch.cat([bucket_pos_w, bucket_quat_w[:, [1, 2, 3, 0]]], dim=-1)  # -> (x, y, z, w)
        self._bucket_xform_wp.assign(wp.from_torch(xf, dtype=wp.transform))

        self._bucket_fill.zero_()
        wp.launch(
            bucket_fill_kernel,
            dim=(num_envs, n_per_env),
            inputs=[
                pos_wp,
                mass_wp,
                self._bucket_xform_wp,
                wp.vec3(*self.cfg.bucket_cavity_lo),
                wp.vec3(*self.cfg.bucket_cavity_hi),
                self._bucket_fill_wp,
            ],
            device=self.device,
        )

    def _get_observations(self) -> dict:
        self._update_soil_observations()

        d = self.rover.data
        base_lin_vel_b = d.root_lin_vel_b
        base_ang_vel_b = d.root_ang_vel_b
        proj_gravity_b = d.projected_gravity_b

        joint_pos = d.joint_pos - d.default_joint_pos
        joint_vel = d.joint_vel

        # bucket pose expressed in the base frame
        bucket_pos_b, bucket_quat_b = subtract_frame_transforms(
            d.root_pos_w, d.root_quat_w,
            d.body_pos_w[:, self._bucket_id], d.body_quat_w[:, self._bucket_id],
        )

        proprio = torch.cat(
            [
                base_lin_vel_b, base_ang_vel_b, proj_gravity_b,
                joint_pos, joint_vel,
                bucket_pos_b, bucket_quat_b,
                self._actions,
            ],
            dim=-1,
        )
        obs = torch.cat([proprio, self._height_map, self._bucket_fill.unsqueeze(-1)], dim=-1)
        return {"policy": obs}

    # ---------------------------------------------------------------- rewards

    def _get_rewards(self) -> torch.Tensor:
        # STUB: shapes are right, the terms are placeholders. Each returns
        # (num_envs,). Tune weights in cfg, not here.
        r_fill = self._reward_fill_delta()
        r_lift = self._reward_lift()
        r_effort = torch.sum(self.rover.data.applied_torque[:, self._arm_ids] ** 2, dim=-1)
        r_rate = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)

        return (
            self.cfg.rew_scale_fill * r_fill
            + self.cfg.rew_scale_lift * r_lift
            + self.cfg.rew_scale_effort * r_effort
            + self.cfg.rew_scale_action_rate * r_rate
            + self.cfg.rew_scale_alive
        )

    def _reward_fill_delta(self) -> torch.Tensor:
        """STUB: mass gained in the bucket since the last step, clipped at zero
        so spilling isn't rewarded by re-scooping."""
        delta = (self._bucket_fill - self._prev_fill).clamp(min=0.0)
        self._prev_fill[:] = self._bucket_fill
        return delta

    def _reward_lift(self) -> torch.Tensor:
        """STUB: fill mass times bucket height above the soil surface. Rewards
        actually lifting the load, not just parking the bucket in the bed."""
        bucket_z = self.rover.data.body_pos_w[:, self._bucket_id, 2] - self.scene.env_origins[:, 2]
        return self._bucket_fill * (bucket_z - self.cfg.scene.soil.spawn.size[2]).clamp(min=0.0)

    # ------------------------------------------------------------------ dones

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        fallen = self.rover.data.projected_gravity_b[:, 2] > -self.cfg.max_tilt_cos
        # STUB: "success" = enough mass scooped and lifted clear of the bed
        bucket_z = self.rover.data.body_pos_w[:, self._bucket_id, 2] - self.scene.env_origins[:, 2]
        succeeded = (self._bucket_fill > self.cfg.fill_success_kg) & (
            bucket_z > self.cfg.scene.soil.spawn.size[2] + 0.2
        )
        terminated = fallen | succeeded
        return terminated, time_out

    # ------------------------------------------------------------------ reset

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.rover._ALL_INDICES
        super()._reset_idx(env_ids)

        # -- rover: default root pose at env origin, default joints
        root_state = self.rover.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.rover.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.rover.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        joint_pos = self.rover.data.default_joint_pos[env_ids]
        joint_vel = self.rover.data.default_joint_vel[env_ids]
        self.rover.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # -- soil: particles back to the spawn grid AND solver history wiped.
        # VERIFY: the MPMObject reset API. Whatever it is, it must also clear
        # elastic strain / plastic Jp / stress (solver.reset in Newton terms),
        # or the bed carries the last episode's stress field into this one and
        # you get a phantom force spike on step 0. This is the reason
        # separate_worlds=True is set: a shared grid can't reset a subset.
        self.soil.reset(env_ids)

        # -- buffers
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._arm_targets[env_ids] = self.rover.data.default_joint_pos[env_ids][:, self._arm_ids]
        self._bucket_fill[env_ids] = 0.0
        self._prev_fill[env_ids] = 0.0
