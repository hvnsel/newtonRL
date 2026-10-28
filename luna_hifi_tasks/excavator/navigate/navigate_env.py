# navigate_env.py
#
# Drive to a goal pose over already-worked terrain and hold it. Actions are
# [forward, yaw]; the arms are held stowed by a fixed position target and the
# drums are held still.
#
# Step order in DirectRLEnv is: _pre_physics_step -> _apply_action (x
# decimation) -> _get_dones -> _get_rewards -> _reset_idx -> _get_observations,
# so what a reward needs from before this step (previous distance, previous
# pose potential, previous action) is updated inside _get_rewards and
# re-seeded in _reset_idx.

from __future__ import annotations

import math

import torch

from ..excavator import TRACK, WHEEL_RADIUS
from ..excavator_cfg import MAX_WHEEL_SPEED
from ..excavator_env_base import ExcavatorEnvBase
from ..mdp import rewards as R
from ..mdp.terrain import degrade_scan, nav_far_pattern, nav_near_pattern, scan_from_raycaster
from ..mdp.observations import NAV_FAR_IMAGE, NAV_NEAR_IMAGE, navigate_obs_spec
from ..terrain_cfg import GOAL_PATCHES
from .navigate_env_cfg import NAV_CRITIC, SCANNER_HEIGHT, ExcavatorNavigateEnvCfg


class ExcavatorNavigateEnv(ExcavatorEnvBase):
    cfg: ExcavatorNavigateEnvCfg

    def __init__(self, cfg: ExcavatorNavigateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        E, dev = self.num_envs, self.device
        self._actions = torch.zeros(E, 2, device=dev)
        self._prev_actions = torch.zeros(E, 2, device=dev)

        # goal pose in the world frame
        self._goal_pos_w = torch.zeros(E, 2, device=dev)
        self._goal_yaw = torch.zeros(E, device=dev)
        self._prev_dist = torch.zeros(E, device=dev)
        self._start_dist = torch.zeros(E, device=dev)
        self._prev_pose_pot = torch.zeros(E, device=dev)
        # Heading error on the last step, for the episode log.
        self._last_herr = torch.zeros(E, device=dev)

        # Arrival. _in_box: inside the pose tolerances this step. _held: steps
        # in a row inside them and stopped. _reached: the hold completed on
        # this step, which fires once an episode. _arrived: it has completed.
        # All set in _get_dones, which runs before _get_rewards in the same
        # step.
        self._in_box = torch.zeros(E, dtype=torch.bool, device=dev)
        self._held = torch.zeros(E, dtype=torch.long, device=dev)
        self._reached = torch.zeros(E, dtype=torch.bool, device=dev)
        self._arrived = torch.zeros(E, dtype=torch.bool, device=dev)
        # Got within curriculum_dist_tol and slowed at some point this
        # episode: what the curriculum counts as a success.
        self._near = torch.zeros(E, dtype=torch.bool, device=dev)
        self._failed_now = torch.zeros(E, dtype=torch.bool, device=dev)

        # curriculum state
        self._goal_dist_hi = float(cfg.goal_dist_range[1])
        self._success_hist = torch.zeros(cfg.curriculum_window, device=dev)
        self._success_ptr = 0
        self._success_n = 0

        # No soil on the rigid tier. Zero, and holds the observation layout
        # identical to the MPM tier.
        self._drum_fill = torch.zeros(E, 2, device=dev)

        # Local offsets each scan was taken at, which degrade_scan scales
        # noise and dropout by.
        self._far_pattern = nav_far_pattern(dev)
        self._near_pattern = nav_near_pattern(dev)

        # Playback only: steps still to run after arriving.
        self._hold_left = torch.zeros(E, dtype=torch.long, device=dev)

        self._trace = None
        self._trace_step = 0
        if cfg.goal_trace_envs > 0:
            self._trace_n = min(int(cfg.goal_trace_envs), E)
            # Line buffered, so a run killed with Ctrl+C still leaves usable
            # rows rather than an empty file.
            self._trace = open(cfg.goal_trace_path, "w", buffering=1)
            self._trace.write(
                "step,ep_step,env,x,y,yaw,goal_x,goal_y,dist,herr,speed_fwd,speed,"
                "reached,timeout\n"
            )
            print(f"[trace] writing {self._trace_n} env(s) to {cfg.goal_trace_path}")

        self._log = R.TermLogger(
            ["progress", "bearing", "goal", "pose", "settle", "fail", "upright", "slip",
             "action_rate", "energy", "time", "fill_change"],
            E, dev,
        )
        self._obs_spec = navigate_obs_spec()
        print(self._obs_spec.describe())
        print(f"[navigate] actor images: far_scan {NAV_FAR_IMAGE}, near_scan {NAV_NEAR_IMAGE}")
        print(NAV_CRITIC.describe())
        rows, cols, spots = self._goal_spots.shape[:3]
        print(f"[navigate] goals from {spots} flat-patch spots per cell, {rows} x {cols} cells")

    # ------------------------------------------------------------------

    def _setup_scene(self):
        super()._setup_scene()
        self.far_scanner = self.scene["far_scanner"]
        self.near_scanner = self.scene["near_scanner"]
        self.terrain = self.scene.terrain
        # _record_outcomes moves envs up and down the rows through
        # update_env_origins, which acts on the terrain_levels a generator
        # terrain with curriculum=True builds.
        if getattr(self.terrain, "terrain_levels", None) is None:
            raise RuntimeError(
                "the terrain importer has no terrain_levels, so difficulty cannot "
                "advance; the navigate scene needs terrain_type='generator' with "
                "curriculum=True on the generator cfg"
            )
        # Goal spots, (rows, cols, spots, 2) in the world frame: the generator
        # adds each cell's origin to the spots it found in that cell.
        patches = self.terrain.flat_patches or {}
        if GOAL_PATCHES not in patches:
            raise RuntimeError(
                f"the terrain has no '{GOAL_PATCHES}' flat patches to draw goals from; "
                "every sub-terrain in terrain_cfg.py needs flat_patch_sampling with "
                f"that key. Found: {sorted(patches)}"
            )
        self._goal_spots = patches[GOAL_PATCHES][..., :2]

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        s = self.cfg.action_smoothing
        self._prev_actions[:] = self._actions
        self._actions[:] = (1.0 - s) * actions.clamp(-1.0, 1.0) + s * self._actions
        # The handover: once an env has arrived its commands stay at zero
        # until it resets. Only reachable with goal_hold_steps > 0; otherwise
        # the episode ends on the step it arrives.
        self._actions[self._arrived] = 0.0

    def _apply_action(self):
        self._apply_drive(self._actions[:, 0], self._actions[:, 1])
        self._apply_arms(torch.full((self.num_envs,), self.cfg.arm_hold_angle, device=self.device))
        self._apply_drums(torch.zeros(self.num_envs, device=self.device))
        # Inlet turned up and held there.
        self._apply_shrouds(torch.full((self.num_envs,), math.pi, device=self.device))

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    def _goal_in_body(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(goal_vec_b (E,2), dist (E,), heading (sin,cos) (E,2), heading
        error (E,) in (-pi, pi])."""
        pos = self.robot.data.root_pos_w.torch[:, :2]
        yaw = self._base_yaw()
        vec_w = self._goal_pos_w - pos
        vec_b = R.world_to_body_xy(vec_w, yaw)
        dist = torch.linalg.norm(vec_b, dim=-1)
        heading = R.heading_error_sin_cos(self._goal_yaw, yaw)
        return vec_b, dist, heading, torch.atan2(heading[:, 0], heading[:, 1])

    def _pose_potential(self, dist: torch.Tensor, herr: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return R.pose_potential(
            dist, herr, c.w_pos_fine, c.pos_sigma, c.w_heading, c.heading_sigma, c.heading_radius
        )

    def _terrain_scan(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(far, near), both chassis-relative heights in the yaw frame."""
        out = []
        for scanner in (self.far_scanner, self.near_scanner):
            d = scanner.data
            out.append(scan_from_raycaster(
                d.pos_w.torch, d.ray_hits_w.torch, SCANNER_HEIGHT, clip=self.cfg.scan_clip
            ))
        return out[0], out[1]

    def _get_observations(self) -> dict:
        p = self._proprio()
        c = self.cfg
        vec_b, dist, heading, _ = self._goal_in_body()
        far, near = self._terrain_scan()

        def degrade(scan, pattern):
            return degrade_scan(
                scan, pattern, c.scan_noise_std, c.scan_dropout,
                c.scan_range_ref, c.scan_invalid,
            )

        parts = {
            "base_lin_vel": p["base_lin_vel"],
            "base_ang_vel": p["base_ang_vel"],
            "projected_gravity": p["projected_gravity"],
            "goal_vec_b": vec_b / c.goal_dist_max,
            "goal_vec_near": vec_b / dist.clamp_min(c.goal_near_range).unsqueeze(-1),
            "goal_heading": heading,
            "wheel_vel": p["wheel_vel"],
            "arm_pos": p["arm_pos"],
            "drum_fill": self._drum_fill,
            "last_action": self._actions,
        }
        policy = self._obs_spec.assemble(parts)

        wheel_vel = self.robot.data.joint_vel.torch
        v_fwd = p["base_lin_vel"][:, 0]
        slip = torch.stack([
            R.wheel_slip(wheel_vel[:, self._left_ids], v_fwd, WHEEL_RADIUS),
            R.wheel_slip(wheel_vel[:, self._right_ids], v_fwd, WHEEL_RADIUS),
        ], dim=-1)
        critic = NAV_CRITIC.assemble({
            "terrain_scan_true": torch.cat([far, near], dim=-1),
            "drum_fill_mass": self._drum_fill,
            "base_lin_vel_w": self.robot.data.root_lin_vel_w.torch,
            "slip": slip,
        })
        n = self.num_envs
        return {
            "policy": policy,
            # Images for the actor's CNN encoders; see NAV_FAR_IMAGE.
            "far_scan": degrade(far, self._far_pattern).reshape(n, *NAV_FAR_IMAGE),
            "near_scan": degrade(near, self._near_pattern).reshape(n, *NAV_NEAR_IMAGE),
            "critic": critic,
        }

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        c, L = self.cfg, self._log
        p = self._proprio()
        vec_b, dist, _, herr = self._goal_in_body()
        d = self.robot.data

        wheel_vel = d.joint_vel.torch[:, self._wheel_ids]
        torque = d.applied_torque.torch[:, self._wheel_ids]
        speed = torch.linalg.norm(p["base_lin_vel"][:, :2], dim=-1)
        pose_pot = self._pose_potential(dist, herr)

        reward = L.add_all({
            "progress": c.w_progress * R.progress_reward(self._prev_dist, dist),
            "bearing": c.w_bearing * R.fade_in(dist, *c.bearing_fade) * R.bearing_alignment(
                vec_b, p["base_lin_vel"][:, 0], MAX_WHEEL_SPEED * WHEEL_RADIUS
            ),
            "goal": c.w_goal * self._reached.float(),
            # Weighted inside the potential; see pose_potential.
            "pose": pose_pot - self._prev_pose_pot,
            "settle": -c.w_settle * self._in_box.float() * R.settle_penalty(
                speed, p["base_ang_vel"][:, 2], 0.5 * TRACK
            ),
            # Every other penalty is per-step, so ending an episode early
            # avoids the rest of them. This is the one-time charge that makes
            # tipping over cost more than the stream it escapes.
            "fail": -c.w_fail * self._failed_now.float(),
            "upright": -c.w_upright * R.upright_penalty(p["projected_gravity"]),
            "slip": -c.w_slip * R.wheel_slip(wheel_vel, p["base_lin_vel"][:, 0], WHEEL_RADIUS),
            "action_rate": -c.w_action_rate * R.action_rate_penalty(self._actions, self._prev_actions),
            "energy": -c.w_energy * R.energy_penalty(torque, wheel_vel),
            "time": -c.w_time * torch.ones_like(dist),
            # No soil on this tier, so this reads zero. An assertion in
            # reward form.
            "fill_change": -c.w_fill_change * torch.zeros_like(dist),
        })
        self._prev_dist[:] = dist
        self._prev_pose_pot[:] = pose_pot
        return reward

    # ------------------------------------------------------------------
    # terminations
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._failed_now = self._failed()
        # Arrival is decided here, not in _get_rewards: DirectRLEnv.step calls
        # dones first, so a flag set by the reward would terminate a step late
        # and _get_rewards would fire again on that extra step, paying w_goal
        # twice.
        _, dist, _, herr = self._goal_in_body()
        d = self.robot.data
        speed = torch.linalg.norm(d.root_lin_vel_b.torch[:, :2], dim=-1)
        yaw_rate = d.root_ang_vel_b.torch[:, 2]
        self._in_box = R.goal_pose_ok(dist, herr, c.goal_dist_tol, c.goal_heading_tol_rad)
        still = R.goal_reached(
            dist, herr, speed, yaw_rate,
            c.goal_dist_tol, c.goal_heading_tol_rad, c.goal_speed_tol, c.goal_yaw_rate_tol,
        )
        self._held = torch.where(still, self._held + 1, torch.zeros_like(self._held))
        self._reached = (self._held >= c.arrive_hold_steps) & ~self._arrived
        self._arrived |= self._reached
        self._near |= (dist < c.curriculum_dist_tol) & (speed < c.curriculum_speed_tol)
        self._last_herr[:] = herr
        if self._trace is not None:
            self._write_trace(dist, herr, time_out)

        done = self._failed_now | self._reached
        if c.goal_hold_steps > 0:
            # Run goal_hold_steps more steps after arriving, with the commands
            # latched to zero by _pre_physics_step, so playback shows the
            # machine left where navigate put it. _reached still fires on the
            # arriving step alone, so the reward, the curriculum and
            # Episode/reached keep their meaning.
            self._hold_left = torch.where(
                self._reached,
                torch.full_like(self._hold_left, c.goal_hold_steps),
                torch.where(self._arrived, self._hold_left - 1, self._hold_left),
            )
            done = self._failed_now | (self._arrived & (self._hold_left <= 0))
        return done, time_out

    def _write_trace(self, dist: torch.Tensor, herr: torch.Tensor, time_out: torch.Tensor) -> None:
        """One CSV row per env per step, in the env frame.

        Positions are env-relative so several envs are comparable; _goal_pos_w
        is in world coordinates, hence the same subtraction. herr is the
        heading error in degrees; reached is 1 from the step the hold
        completes.
        """
        n = self._trace_n
        o = self.scene.env_origins[:n, :2]
        pos = (self.robot.data.root_pos_w.torch[:n, :2] - o).tolist()
        goal = (self._goal_pos_w[:n] - o).tolist()
        yaw = self._base_yaw()[:n].tolist()
        vel = self.robot.data.root_lin_vel_b.torch[:n, :2]
        spd = vel[:, 0].tolist()
        # What goal_reached actually gates on: the planar magnitude.
        mag = torch.linalg.norm(vel, dim=-1).tolist()
        d = dist[:n].tolist()
        h = torch.rad2deg(herr[:n]).tolist()
        reached = self._arrived[:n].tolist()
        tout = time_out[:n].tolist()
        ep = self.episode_length_buf[:n].tolist()
        for i in range(n):
            self._trace.write(
                f"{self._trace_step},{ep[i]},{i},"
                f"{pos[i][0]:.4f},{pos[i][1]:.4f},{yaw[i]:.4f},"
                f"{goal[i][0]:.4f},{goal[i][1]:.4f},"
                f"{d[i]:.4f},{h[i]:.3f},{spd[i]:.4f},{mag[i]:.4f},"
                f"{int(reached[i])},{int(tout[i])}\n"
            )
        self._trace_step += 1

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def _record_outcomes(self, env_ids: torch.Tensor) -> None:
        """Move each env's terrain level, and widen the goal band once the
        success rate over the window clears the threshold. Success here is
        getting near the goal (see curriculum_dist_tol), not the held
        arrival.

        The ring buffer is written with one scatter, since a reset batch can
        carry every env.
        """
        c = self.cfg
        k = env_ids.numel()
        if k == 0:
            return

        near = self._near[env_ids]
        # Did not get near and closed less than terrain_demote_fraction of
        # the gap: this env moves down a level.
        stalled = ~near & (self._prev_dist[env_ids] > c.terrain_demote_fraction * self._start_dist[env_ids])
        self.terrain.update_env_origins(env_ids, near, stalled)

        idx = (self._success_ptr + torch.arange(k, device=self.device)) % c.curriculum_window
        self._success_hist[idx] = near.float()
        self._success_ptr = int((self._success_ptr + k) % c.curriculum_window)
        self._success_n = min(self._success_n + k, c.curriculum_window)

        if self._success_n >= c.curriculum_window:
            rate = float(self._success_hist.mean())
            if rate > c.curriculum_success_rate and self._goal_dist_hi < c.goal_dist_max:
                self._goal_dist_hi = min(self._goal_dist_hi + c.curriculum_step, c.goal_dist_max)
                self._success_n = 0

    def _sample_goals(self, env_ids: torch.Tensor, spawn_xy: torch.Tensor) -> None:
        """Goals from the flat-patch spots of the cell each env is now in,
        measured from the spawn position the reset just wrote; see
        _reset_robot on why root_pos_w is not read here.

        A distance is drawn from the curriculum band and the spot whose
        distance from the spawn comes closest to it is taken. Spots nearer
        than the band's low end lose to any other. With 512 spots across the
        cell the winner is usually within a decimetre of the drawn distance,
        and which spot on that ring wins is as good as a random bearing.
        """
        c = self.cfg
        n = env_ids.numel()
        dev = self.device

        spots = self._goal_spots[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]
        ]                                                                   # (n, P, 2)
        reach = torch.linalg.norm(spots - spawn_xy.unsqueeze(1), dim=-1)   # (n, P)
        lo = c.goal_dist_range[0]
        want = lo + (self._goal_dist_hi - lo) * torch.rand(n, 1, device=dev)
        cost = (reach - want).abs() + (reach < lo).float() * 1.0e3
        goal = spots[torch.arange(n, device=dev), cost.argmin(dim=1)]
        self._goal_pos_w[env_ids] = goal

        # Arrive facing roughly the way the machine drove in, which is what a
        # dig approach looks like. goal_yaw_random_frac of episodes draw a
        # heading uniformly instead.
        rel = goal - spawn_xy
        bearing = torch.atan2(rel[:, 1], rel[:, 0])
        yaw = bearing + torch.randn(n, device=dev) * c.goal_yaw_spread
        free = torch.rand(n, device=dev) < c.goal_yaw_random_frac
        self._goal_yaw[env_ids] = torch.where(
            free, (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi, yaw
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Read before super() clears the episode buffers.
        exits = self._exit_stats(env_ids)
        exits["Episode/reached"] = float(self._arrived[env_ids].float().mean())
        exits["Episode/near"] = float(self._near[env_ids].float().mean())
        exits["Episode/failed"] = float(self._failed_now[env_ids].float().mean())
        exits["Episode/end_dist"] = float(self._prev_dist[env_ids].mean())
        exits["Episode/end_heading_deg"] = float(torch.rad2deg(self._last_herr[env_ids].abs()).mean())
        # Reads last episode's outcome, before it is cleared below.
        self._record_outcomes(env_ids)
        super()._reset_idx(env_ids)

        n = env_ids.numel()
        yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * math.pi
        # Scattered inside the terrain cell, so envs sharing an origin see
        # different ground.
        j = self.cfg.spawn_xy_jitter
        xy_off = (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * j if j > 0.0 else None
        spawn_xy = self._reset_robot(env_ids, yaw, self.cfg.arm_hold_angle, xy_off)

        self._sample_goals(env_ids, spawn_xy)
        # Initial distance and pose potential from the written spawn.
        d0 = torch.linalg.norm(self._goal_pos_w[env_ids] - spawn_xy, dim=-1)
        e0 = self._goal_yaw[env_ids] - yaw
        self._prev_dist[env_ids] = d0
        self._start_dist[env_ids] = d0.clamp_min(1e-3)
        self._prev_pose_pot[env_ids] = self._pose_potential(d0, torch.atan2(torch.sin(e0), torch.cos(e0)))
        self._last_herr[env_ids] = 0.0
        self._in_box[env_ids] = False
        self._held[env_ids] = 0
        self._reached[env_ids] = False
        self._arrived[env_ids] = False
        self._near[env_ids] = False
        self._hold_left[env_ids] = 0
        self._failed_now[env_ids] = False
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        self.extras["log"] = self._log.flush(env_ids)
        self.extras["log"].update(exits)
        self.extras["log"]["Curriculum/goal_dist_hi"] = self._goal_dist_hi
        self.extras["log"]["Curriculum/terrain_level"] = float(
            self.terrain.terrain_levels.float().mean()
        )
