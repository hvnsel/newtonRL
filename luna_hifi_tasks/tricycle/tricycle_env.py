# tricycle_env.py
#
# The smallest DirectRLEnv that exercises the whole stack: MJCF articulation,
# MPM soil, two-way contact, batched reset, rsl_rl. Reward is per-step forward
# displacement, which sums to total distance over the 2 s episode.

# proof of concept we can have an RL environment with an articulated vehicle AND MPM soil

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from isaaclab_newton.assets import MPMObject

from .tricycle import JOINT_REAR, JOINT_STEER
from .tricycle_env_cfg import TricycleEnvCfg


class TricycleEnv(DirectRLEnv):
    cfg: TricycleEnvCfg

    # This is a child class of DirectRLEnv which is a child class of InteractiveSceneEnv.
    def __init__(self, cfg: TricycleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._steer_id, _ = self.car.find_joints([JOINT_STEER])
        self._rear_ids, _ = self.car.find_joints(JOINT_REAR)
        # one steerable front wheel and two rear wheels
        assert len(self._steer_id) == 1 and len(self._rear_ids) == 2

        # the actions tensor is 2D: number of environments x 2 (throttle, steer)
        self._actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_x = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        # InteractiveScene has already spawned everything in TricycleSceneCfg.
        # Constructing assets here would try to spawn them a second time.
        self.car: Articulation = self.scene["tricycle"]
        self.soil: MPMObject = self.scene["soil"]


    '''
    DirectRLEnv has a step function that does the following:
    This function performs the following steps:
        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.
    We must implement the following abstract methods in this class:
    '''
    def _pre_physics_step(self, actions: torch.Tensor):
        # clamp our actions into [-1, 1]
        self._actions[:] = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        throttle = self._actions[:, 0:1] * self.cfg.max_wheel_speed
        steer = self._actions[:, 1:2] * self.cfg.max_steer
        self.car.set_joint_velocity_target(throttle.expand(-1, 2), joint_ids=self._rear_ids)
        self.car.set_joint_position_target(steer, joint_ids=self._steer_id)

    def _get_observations(self) -> dict:
        d = self.car.data
        obs = torch.cat(
            [
                d.root_lin_vel_b,
                d.root_ang_vel_b,
                d.projected_gravity_b,
                d.joint_pos[:, self._steer_id],
                d.joint_vel[:, self._steer_id],
                d.joint_vel[:, self._rear_ids],
                self._actions,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        x = self.car.data.root_pos_w[:, 0] - self.scene.env_origins[:, 0]
        dx = torch.nan_to_num(x - self._prev_x, nan=0.0, posinf=0.0, neginf=0.0)
        self._prev_x[:] = x
        return dx.clamp(-0.1, 0.1)     # 0.1 m per 20 ms step = 5 m/s ceiling

    # find out which envs are done, either by timeout or by invalid state
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        pos = self.car.data.root_pos_w - self.scene.env_origins
        vel = self.car.data.root_lin_vel_b

        invalid = ~torch.isfinite(pos).all(dim=-1) | ~torch.isfinite(vel).all(dim=-1)
        invalid |= pos.abs().max(dim=-1).values > 50.0
        invalid |= vel.abs().max(dim=-1).values > 50.0

        off_strip = pos[:, 1].abs() > self.cfg.max_lateral_drift
        flipped = self.car.data.projected_gravity_b[:, 2] < self.cfg.flip_gravity_z

        return invalid | off_strip | flipped, time_out

    #  Reset the envs that are done, and reset the episode length counter for those envs.
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        root_state = self.car.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self.car.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.car.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.car.write_joint_state_to_sim(
            self.car.data.default_joint_pos[env_ids],
            self.car.data.default_joint_vel[env_ids],
            None,
            env_ids,
        )

        # Puts every particle of these envs back at its spawn position with zero
        # velocity. Solver-side history (warm starts, plastic strain) is not
        # cleared by this call; NewtonMPMManager.reset_solver_state exists for
        # that, and the tricycle trains fine without it.
        self.soil.reset(env_ids)

        self._actions[env_ids] = 0.0
        self._prev_x[env_ids] = root_state[:, 0] - self.scene.env_origins[env_ids, 0]
