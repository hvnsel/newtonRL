# navigate_env.py
#
# Drive to a goal pose over already-worked terrain. Actions are [forward,
# yaw]; the arms are held stowed by a fixed position target and the drums are
# held still, so the navigator has no actuator with which to disturb soil.
#
# Step order in DirectRLEnv is: _pre_physics_step -> _apply_action (x
# decimation) -> _get_dones -> _get_rewards -> _reset_idx -> _get_observations.
# Anything a reward needs from "before this step" (previous distance, previous
# action) is therefore updated inside _get_rewards and re-seeded in _reset_idx.

from __future__ import annotations

import math

import torch

from ..excavator import WHEEL_RADIUS
from ..excavator_cfg import MAX_WHEEL_SPEED
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
        self._start_dist = torch.zeros(E, device=dev)
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
        self.terrain = self.scene.terrain
        # max_init_terrain_level puts envs on rows 0-2 of 6 and nothing else
        # moves them, so without this call the three hardest rows are built at
        # startup and never driven on.
        # update_env_origins exists on every TerrainImporter but only does
        # anything when terrain_levels was built, which needs a generator
        # terrain with curriculum=True.
        if getattr(self.terrain, "terrain_levels", None) is None:
            raise RuntimeError(
                "the terrain importer has no terrain_levels, so difficulty cannot "
                "advance; the navigate scene needs terrain_type='generator' with "
                "curriculum=True on the generator cfg"
            )

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
        # Inlet turned up and held there. The navigator has no actuator with
        # which to load a drum, the same way it has none to lower an arm.
        self._apply_shrouds(torch.full((self.num_envs,), math.pi, device=self.device))

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
            + c.w_bearing * L.add(
                "bearing",
                R.bearing_alignment(vec_b, p["base_lin_vel"][:, 0], MAX_WHEEL_SPEED * WHEEL_RADIUS),
            )
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
        """Move each env's terrain level, and widen the goal band once the
        success rate over the window clears the threshold.

        The ring buffer is written with one scatter. Assigning element by
        element costs a device sync per episode, and up to 1024 of them land
        in a single reset batch.
        """
        c = self.cfg
        k = env_ids.numel()
        if k == 0:
            return

        reached = self._reached[env_ids]
        # Closed less than terrain_demote_fraction of the gap and did not
        # arrive: this env gets easier ground.
        stalled = ~reached & (self._prev_dist[env_ids] > c.terrain_demote_fraction * self._start_dist[env_ids])
        self.terrain.update_env_origins(env_ids, reached, stalled)

        idx = (self._success_ptr + torch.arange(k, device=self.device)) % c.curriculum_window
        self._success_hist[idx] = reached.float()
        self._success_ptr = int((self._success_ptr + k) % c.curriculum_window)
        self._success_n = min(self._success_n + k, c.curriculum_window)

        if self._success_n >= c.curriculum_window:
            rate = float(self._success_hist.mean())
            if rate > c.curriculum_success_rate and self._goal_dist_hi < c.goal_dist_max:
                self._goal_dist_hi = min(self._goal_dist_hi + c.curriculum_step, c.goal_dist_max)
                self._success_n = 0

    def _sample_goals(self, env_ids: torch.Tensor, spawn_xy: torch.Tensor) -> None:
        """Goals centred on the spawn position the reset just WROTE. Reading
        root_pos_w here would return the pre-reset pose (see _reset_robot).

        The goal stays inside the sub-terrain this env was assigned: the
        bearing is free, and the distance is capped where that ray leaves the
        cell. Sampling a distance first and clamping the point afterwards
        piles goals onto the boundary instead.
        """
        c = self.cfg
        n = env_ids.numel()
        dev = self.device
        half = c.goal_cell_half

        bearing = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi
        dx, dy = torch.cos(bearing), torch.sin(bearing)
        rel = spawn_xy - self.scene.env_origins[env_ids, :2]

        # Distance along (dx, dy) from the spawn to the cell boundary.
        eps = 1e-6
        tx = (torch.where(dx > 0, half - rel[:, 0], -half - rel[:, 0])
              / torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx))
        ty = (torch.where(dy > 0, half - rel[:, 1], -half - rel[:, 1])
              / torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy))
        reach = torch.minimum(tx, ty).clamp_min(0.0)

        lo = c.goal_dist_range[0]
        hi = torch.clamp(reach, max=self._goal_dist_hi)
        dist = torch.minimum(lo + (hi - lo).clamp_min(0.0) * torch.rand(n, device=dev), reach)

        self._goal_pos_w[env_ids, 0] = spawn_xy[:, 0] + dist * dx
        self._goal_pos_w[env_ids, 1] = spawn_xy[:, 1] + dist * dy

        # Arrive facing roughly the way you drove in, which is what a dig
        # approach looks like. A uniform heading made half of all episodes end
        # in a large turn on the spot, unrelated to navigating.
        yaw = bearing + torch.randn(n, device=dev) * c.goal_yaw_spread
        free = torch.rand(n, device=dev) < c.goal_yaw_random_frac
        self._goal_yaw[env_ids] = torch.where(
            free, (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi, yaw
        )

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
        d0 = torch.linalg.norm(self._goal_pos_w[env_ids] - spawn_xy, dim=-1)
        self._prev_dist[env_ids] = d0
        self._start_dist[env_ids] = d0.clamp_min(1e-3)
        self._reached[env_ids] = False
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        self.extras["log"] = self._log.flush(env_ids, self.cfg.episode_length_s)
        self.extras["log"]["Curriculum/goal_dist_hi"] = self._goal_dist_hi
        self.extras["log"]["Curriculum/terrain_level"] = float(
            self.terrain.terrain_levels.float().mean()
        )
