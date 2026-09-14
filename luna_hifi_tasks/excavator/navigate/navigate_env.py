# navigate_env.py
#
# Drive to a goal pose over already-worked terrain. Actions are [forward,
# yaw]; the arms are held stowed by a fixed position target and the drums are
# held still, so the navigator physically cannot disturb soil. That is the
# whole mechanism keeping navigation and excavation mutually exclusive -- a
# missing actuator, not a penalty the policy could trade against.
#
# Step order in DirectRLEnv is: _pre_physics_step -> _apply_action (x
# decimation) -> _get_dones -> _get_rewards -> _reset_idx -> _get_observations.
# Anything a reward needs from "before this step" (previous distance, previous
# action) is therefore updated inside _get_rewards and re-seeded in _reset_idx.

from __future__ import annotations

import math

import torch

from ..excavator import WHEEL_RADIUS
from ..excavator_env_base import ExcavatorEnvBase
from ..mdp import rewards as R
from ..mdp.terrain import scan_from_raycaster
from .navigate_env_cfg import NAV_CRITIC, NAV_OBS, SCANNER_HEIGHT, ExcavatorNavigateEnvCfg


class ExcavatorNavigateEnv(ExcavatorEnvBase):
    cfg: ExcavatorNavigateEnvCfg

    def __init__(self, cfg: ExcavatorNavigateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        E, dev = self.num_envs, self.device
        self._actions = torch.zeros(E, 2, device=dev)
        self._prev_actions = torch.zeros(E, 2, device=dev)

        # goal pose in WORLD frame (so it survives the machine moving)
        self._goal_pos_w = torch.zeros(E, 2, device=dev)
        self._goal_yaw = torch.zeros(E, device=dev)
        self._prev_dist = torch.zeros(E, device=dev)
        self._reached = torch.zeros(E, dtype=torch.bool, device=dev)

        # curriculum state
        self._goal_dist_hi = float(cfg.goal_dist_range[1])
        self._success_hist = torch.zeros(cfg.curriculum_window, device=dev)
        self._success_ptr = 0
        self._success_n = 0

        # on the rigid tier there is no soil; these stay zero but keep the
        # observation layout identical to the MPM tier
        self._drum_fill = torch.zeros(E, 2, device=dev)

        self._log = R.TermLogger(
            ["progress", "bearing", "goal", "upright", "slip", "action_rate", "energy", "time", "fill_change"],
            E, dev,
        )
        print(NAV_OBS.describe())
        print(NAV_CRITIC.describe())

    # ------------------------------------------------------------------

    def _setup_scene(self):
        super()._setup_scene()
        self.scanner = self.scene["height_scanner"]

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        s = self.cfg.action_smoothing
        self._prev_actions[:] = self._actions
        self._actions[:] = (1.0 - s) * actions.clamp(-1.0, 1.0) + s * self._actions

    def _apply_action(self):
        self._apply_drive(self._actions[:, 0], self._actions[:, 1])
        self._apply_arms(torch.full((self.num_envs,), self.cfg.arm_hold_angle, device=self.device))
        self._apply_drums(torch.zeros(self.num_envs, device=self.device))

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    def _goal_in_body(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(goal_vec_b (E,2), dist (E,), heading (sin,cos) (E,2))."""
        pos = self.robot.data.root_pos_w.torch[:, :2]
        yaw = self._base_yaw()
        vec_w = self._goal_pos_w - pos
        vec_b = R.world_to_body_xy(vec_w, yaw)
        dist = torch.linalg.norm(vec_b, dim=-1)
        return vec_b, dist, R.heading_error_sin_cos(self._goal_yaw, yaw)

    def _terrain_scan(self) -> torch.Tensor:
        d = self.scanner.data
        return scan_from_raycaster(
            d.pos_w.torch, d.ray_hits_w.torch, SCANNER_HEIGHT, clip=self.cfg.scan_clip
        )

    def _get_observations(self) -> dict:
        p = self._proprio()
        vec_b, dist, heading = self._goal_in_body()
        scan = self._terrain_scan()

        actor_scan = scan
        if self.cfg.scan_noise_std > 0.0:
            actor_scan = scan + torch.randn_like(scan) * self.cfg.scan_noise_std

        policy = NAV_OBS.assemble({
            "base_lin_vel": p["base_lin_vel"],
            "base_ang_vel": p["base_ang_vel"],
            "projected_gravity": p["projected_gravity"],
            "goal_vec_b": vec_b / self.cfg.goal_dist_max,
            "goal_heading": heading,
            "wheel_vel": p["wheel_vel"],
            "arm_pos": p["arm_pos"],
            "drum_fill": self._drum_fill,
            "last_action": self._actions,
            "terrain_scan": actor_scan,
        })

        wheel_vel = self.robot.data.joint_vel.torch
        v_fwd = p["base_lin_vel"][:, 0]
        slip = torch.stack([
            R.wheel_slip(wheel_vel[:, self._left_ids], v_fwd, WHEEL_RADIUS),
            R.wheel_slip(wheel_vel[:, self._right_ids], v_fwd, WHEEL_RADIUS),
        ], dim=-1)
        critic = NAV_CRITIC.assemble({
            "terrain_scan_true": scan,
            "drum_fill_mass": self._drum_fill,
            "base_lin_vel_w": self.robot.data.root_lin_vel_w.torch,
            "slip": slip,
        })
        return {"policy": policy, "critic": critic}

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        c, L = self.cfg, self._log
        p = self._proprio()
        vec_b, dist, heading = self._goal_in_body()
        d = self.robot.data

        self._reached = R.goal_reached(dist, heading[:, 1], c.goal_dist_tol, math.cos(c.goal_heading_tol_rad))

        wheel_vel = d.joint_vel.torch[:, self._wheel_ids]
        torque = d.applied_torque.torch[:, self._wheel_ids]

        reward = (
            c.w_progress * L.add("progress", R.progress_reward(self._prev_dist, dist))
            + c.w_bearing * L.add("bearing", R.bearing_alignment(vec_b))
            + c.w_goal * L.add("goal", self._reached.float())
            - c.w_upright * L.add("upright", R.upright_penalty(p["projected_gravity"]))
            - c.w_slip * L.add("slip", R.wheel_slip(wheel_vel, p["base_lin_vel"][:, 0], WHEEL_RADIUS))
            - c.w_action_rate * L.add("action_rate", R.action_rate_penalty(self._actions, self._prev_actions))
            - c.w_energy * L.add("energy", R.energy_penalty(torque, wheel_vel))
            - c.w_time * L.add("time", torch.ones_like(dist))
            - c.w_fill_change * L.add("fill_change", torch.zeros_like(dist))   # no soil on this tier
        )
        self._prev_dist[:] = dist
        return reward

    # ------------------------------------------------------------------
    # terminations
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        failed = self._failed()
        terminated = failed | self._reached
        return terminated, time_out

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def _record_outcomes(self, env_ids: torch.Tensor) -> None:
        """Feed episode outcomes into the curriculum and widen the goal band
        once the success rate over the window clears the threshold."""
        succ = self._reached[env_ids].float()
        for v in succ:
            self._success_hist[self._success_ptr] = v
            self._success_ptr = (self._success_ptr + 1) % self.cfg.curriculum_window
            self._success_n = min(self._success_n + 1, self.cfg.curriculum_window)
        if self._success_n >= self.cfg.curriculum_window:
            rate = float(self._success_hist.mean())
            if rate > self.cfg.curriculum_success_rate and self._goal_dist_hi < self.cfg.goal_dist_max:
                self._goal_dist_hi = min(self._goal_dist_hi + self.cfg.curriculum_step, self.cfg.goal_dist_max)
                self._success_n = 0

    def _sample_goals(self, env_ids: torch.Tensor, spawn_xy: torch.Tensor) -> None:
        """Goals centred on the spawn position the reset just WROTE. Reading
        root_pos_w here would return the pre-reset pose (see _reset_robot)."""
        n = env_ids.numel()
        dev = self.device
        lo = self.cfg.goal_dist_range[0]
        dist = lo + (self._goal_dist_hi - lo) * torch.rand(n, device=dev)
        bearing = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi
        self._goal_pos_w[env_ids, 0] = spawn_xy[:, 0] + dist * torch.cos(bearing)
        self._goal_pos_w[env_ids, 1] = spawn_xy[:, 1] + dist * torch.sin(bearing)
        self._goal_yaw[env_ids] = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # curriculum bookkeeping uses last episode's outcome, before it is cleared
        self._record_outcomes(env_ids)
        super()._reset_idx(env_ids)

        n = env_ids.numel()
        yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * math.pi
        # scatter inside the terrain cell so envs sharing an origin see
        # different ground
        j = self.cfg.spawn_xy_jitter
        xy_off = (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * j if j > 0.0 else None
        spawn_xy = self._reset_robot(env_ids, yaw, self.cfg.arm_hold_angle, xy_off)

        self._sample_goals(env_ids, spawn_xy)
        # initial distance from the written spawn; world-frame norm is yaw-invariant
        self._prev_dist[env_ids] = torch.linalg.norm(self._goal_pos_w[env_ids] - spawn_xy, dim=-1)
        self._reached[env_ids] = False
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        self.extras["log"] = self._log.flush(env_ids, self.cfg.episode_length_s)
        self.extras["log"]["Curriculum/goal_dist_hi"] = self._goal_dist_hi
