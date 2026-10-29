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

from ..excavator import AXLE_X, TRACK, WHEEL_HALF_W, WHEEL_RADIUS, WHEEL_Y
from ..excavator_cfg import MAX_WHEEL_SPEED
from ..excavator_env_base import ExcavatorEnvBase
from ..mdp import rewards as R
from ..mdp.terrain import degrade_scan, nav_far_pattern, nav_near_pattern, scan_from_raycaster
from ..mdp.observations import NAV_FAR_IMAGE, NAV_NEAR_IMAGE, navigate_obs_spec
from ..terrain_cfg import GOAL_PATCHES, SPAWN_PATCHES
from .navigate_env_cfg import NAV_CRITIC, ExcavatorNavigateEnvCfg

# Playback goal marker: a disc on the goal spot and a bar through it along the
# goal's axis, either end of which the machine may face. 5 m of bar sticks out
# 0.74 m past each drum, 1.76 m from the chassis centre, when the machine is
# parked on the goal.
MARKER_RGB = (0.10, 0.90, 0.20)
MARKER_DISC = (0.5, 0.04)               # radius, height
MARKER_BAR = (5.0, 0.20, 0.04)          # length, width, height
MARKER_LIFT = 0.15                      # m above the goal spot, clear of small relief

# Where the machine meets the ground, body frame. First each tyre's contact
# line, at its middle and both faces: a spawn sets the machine down on the
# highest of these. Then a grid the ground has to be even across: across the
# tyres' outer faces, and lengthwise to 1.25 m, past the wheels' front and
# back at 1.0 m to where the stowed arms' roots rise above the tallest
# boulder (0.6 m, reached 1.18 m out). At 0.25 x 0.34 m no boulder (0.4 m
# radius and up) fits between points, and one reaching in from outside is
# caught 2 cm in. See _draw_spawns.
TYRE_XY = tuple(
    (sx * AXLE_X, sy * WHEEL_Y + dy)
    for sx in (1, -1) for sy in (1, -1) for dy in (-WHEEL_HALF_W, 0.0, WHEEL_HALF_W)
)
_REACH_Y = WHEEL_Y + WHEEL_HALF_W
FOOTPRINT_XY = TYRE_XY + tuple(
    (0.25 * i, 0.5 * j * _REACH_Y) for i in range(-5, 6) for j in range(-2, 3)
)
# Where the spawn ray-casts start, straight down: above any terrain, as in
# Isaac Lab's flat-patch search.
RAY_TOP = 100.0

# Blocked-route goals, see cfg.goal_blocked_frac: how many candidate goals are
# tested per env, the spots whose distance from the spawn is nearest the drawn
# one; the lines along each route the ground is read on, across the machine's
# 1.35 m width; and the samples along each line, 0.25 m apart over the
# longest goal.
GOAL_CANDIDATES = 32
ROUTE_LANES = (-0.6, 0.0, 0.6)
ROUTE_SAMPLES = 73


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
        # Ground height at the goal spot, for the playback marker.
        self._goal_z_w = torch.zeros(E, device=dev)
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

        # Stuck check, see cfg.stuck_window_steps: the last window of positions
        # and headings, as ring buffers, and steps since the reset.
        W = max(int(cfg.stuck_window_steps), 1)
        self._win_pos = torch.zeros(E, W, 2, device=dev)
        self._win_yaw = torch.zeros(E, W, device=dev)
        self._win_steps = torch.zeros(E, dtype=torch.long, device=dev)
        self._stuck = torch.zeros(E, dtype=torch.bool, device=dev)
        # Whether this episode's straight route to its goal is blocked.
        self._route_is_blocked = torch.zeros(E, dtype=torch.bool, device=dev)

        # The terrain as a Warp mesh, for setting spawns down on it; built on
        # the first reset, once the scene is on the stage.
        self._ground = None
        self._footprint = torch.tensor(FOOTPRINT_XY, device=dev)

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

        # The clean scan's mean height and share of cells at -scan_clip on the
        # last step, for the log: a scan that sees the ground reads about the
        # chassis height, +0.3 m, and almost none at the clip; a blind one
        # reads -scan_clip everywhere.
        self._scan_mean = torch.zeros((), device=dev)
        self._scan_at_clip = torch.zeros((), device=dev)

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

        # Playback only; see cfg.goal_marker.
        self._goal_marker = self._make_goal_marker() if cfg.goal_marker else None

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
        print(
            f"[navigate] goals from {spots} and spawns from {self._spawn_spots.shape[2]} "
            f"flat-patch spots per cell, {rows} x {cols} cells"
        )

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
        # Goal and spawn spots, (rows, cols, spots, 3) in the world frame: the
        # generator adds each cell's origin to the spots it found in that cell.
        patches = self.terrain.flat_patches or {}
        for key in (GOAL_PATCHES, SPAWN_PATCHES):
            if key not in patches:
                raise RuntimeError(
                    f"the terrain has no '{key}' flat patches to draw from; every "
                    "sub-terrain in terrain_cfg.py needs flat_patch_sampling with "
                    f"that key. Found: {sorted(patches)}"
                )
        self._goal_spots = patches[GOAL_PATCHES][..., :2]
        # The height the patch search raycast at each spot: the ground there.
        self._goal_spots_z = patches[GOAL_PATCHES][..., 2]
        self._spawn_spots = patches[SPAWN_PATCHES][..., :2]

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
        """(goal_vec_b (E,2), dist (E,), heading (sin,cos) of twice the error
        (E,2), heading error to the goal's axis (E,) in (-pi/2, pi/2]).

        The heading is an axis: facing along the goal heading and facing
        against it are the same pose, since the machine has no front."""
        pos = self.robot.data.root_pos_w.torch[:, :2]
        yaw = self._base_yaw()
        vec_w = self._goal_pos_w - pos
        vec_b = R.world_to_body_xy(vec_w, yaw)
        dist = torch.linalg.norm(vec_b, dim=-1)
        heading = R.axis_heading_sin_cos(self._goal_yaw, yaw)
        return vec_b, dist, heading, 0.5 * torch.atan2(heading[:, 0], heading[:, 1])

    def _pose_potential(self, dist: torch.Tensor, herr: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return R.pose_potential(
            dist, herr, c.w_pos_fine, c.pos_sigma, c.w_heading, c.heading_sigma, c.heading_radius
        )

    def _terrain_scan(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(far, near), both chassis-relative heights in the yaw frame."""
        chassis = self.robot.data.root_pos_w.torch
        out = []
        for scanner in (self.far_scanner, self.near_scanner):
            out.append(scan_from_raycaster(chassis, scanner.data.ray_hits_w.torch, clip=self.cfg.scan_clip))
        return out[0], out[1]

    def _get_observations(self) -> dict:
        # Every step rather than on reset alone: the marker backend attaches to
        # the viewer on the first call after the viewer exists, which can be
        # later than the first reset.
        self._show_goal_marker()
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
        clean = torch.cat([far, near], dim=-1)
        self._scan_mean = clean.mean()
        self._scan_at_clip = (clean <= -c.scan_clip).float().mean()
        critic = NAV_CRITIC.assemble({
            "terrain_scan_true": clean,
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
        if c.stuck_window_steps > 0:
            self._stuck = self._stuck_now(dist)
            self._failed_now |= self._stuck
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

    def _stuck_now(self, dist: torch.Tensor) -> torch.Tensor:
        """Away from the goal, and over the last stuck_window_steps steps
        neither moved stuck_min_move nor turned stuck_min_turn, both net.

        The window slides: each step overwrites the slot it wrote
        stuck_window_steps steps before, and until then the slots hold the
        spawn, so the first test comes a whole window after the reset. Both
        are net changes across the window, so driving back and forth into a
        boulder counts as stuck, and so does steering jitter: training samples
        every action with noise, which summed turning counted as turning. A
        real turn still counts.
        """
        c = self.cfg
        rows = torch.arange(self.num_envs, device=self.device)
        slot = self._win_steps % c.stuck_window_steps
        pos = self.robot.data.root_pos_w.torch[:, :2]
        yaw = self._base_yaw()
        moved = torch.linalg.norm(pos - self._win_pos[rows, slot], dim=-1)
        turn = yaw - self._win_yaw[rows, slot]
        turned = torch.atan2(torch.sin(turn), torch.cos(turn)).abs()
        self._win_pos[rows, slot] = pos
        self._win_yaw[rows, slot] = yaw
        self._win_steps += 1
        return (
            (self._win_steps >= c.stuck_window_steps)
            & (moved < c.stuck_min_move)
            & (turned < c.stuck_min_turn)
            & (dist > c.goal_near_range)
            & ~self._arrived
        )

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
    # playback goal marker
    # ------------------------------------------------------------------

    def _make_goal_marker(self):
        """The goal's VisualizationMarkers, or None if they cannot be built.

        Footage without the marker is still footage, so a failure here is
        reported and playback goes on without it.
        """
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            look = sim_utils.PreviewSurfaceCfg(diffuse_color=MARKER_RGB)
            return VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/navigate_goal",
                markers={
                    "disc": sim_utils.CylinderCfg(
                        radius=MARKER_DISC[0], height=MARKER_DISC[1], visual_material=look
                    ),
                    "bar": sim_utils.CuboidCfg(size=MARKER_BAR, visual_material=look),
                },
            ))
        except Exception as exc:  # noqa: BLE001
            print(f"[navigate] no goal marker, footage will not show the goal: {exc!r}")
            return None

    def _show_goal_marker(self) -> None:
        """A disc on each env's goal spot and a bar through it along the goal's
        axis. Two markers per env, env-major, which is how the Newton viewer
        splits a batch across envs."""
        if self._goal_marker is None:
            return
        n, dev = self.num_envs, self.device
        yaw = self._goal_yaw
        z = (self._goal_z_w + MARKER_LIFT).unsqueeze(-1)
        disc = torch.cat([self._goal_pos_w, z], dim=-1)
        bar = disc
        zero = torch.zeros_like(yaw)
        quat = torch.stack([zero, zero, torch.sin(0.5 * yaw), torch.cos(0.5 * yaw)], dim=-1)    # xyzw
        try:
            self._goal_marker.visualize(
                translations=torch.stack([disc, bar], dim=1).reshape(2 * n, 3),
                orientations=torch.stack([quat, quat], dim=1).reshape(2 * n, 4),
                marker_indices=torch.tensor([0, 1], device=dev).repeat(n),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[navigate] goal marker failed, footage will not show the goal: {exc!r}")
            self._goal_marker = None

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

        For cfg.goal_blocked_frac of envs the winner is instead the nearest
        of the GOAL_CANDIDATES closest spots whose straight route from the
        spawn is blocked (_route_blocked). Where none is, which on the worked
        cells is always, the usual winner stands.
        """
        c = self.cfg
        n = env_ids.numel()
        dev = self.device

        cell = (self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids])
        spots = self._goal_spots[cell]                                      # (n, P, 2)
        reach = torch.linalg.norm(spots - spawn_xy.unsqueeze(1), dim=-1)   # (n, P)
        lo = c.goal_dist_range[0]
        want = lo + (self._goal_dist_hi - lo) * torch.rand(n, 1, device=dev)
        cost = (reach - want).abs() + (reach < lo).float() * 1.0e3
        rows = torch.arange(n, device=dev)
        if c.goal_blocked_frac > 0.0:
            k = min(GOAL_CANDIDATES, spots.shape[1])
            cand_cost, cand = cost.topk(k, dim=1, largest=False)            # nearest first
            ends = torch.gather(spots, 1, cand.unsqueeze(-1).expand(n, k, 2))
            blocked = self._route_blocked(spawn_xy, ends) & (cand_cost < 1.0e3)
            use = (torch.rand(n, device=dev) < c.goal_blocked_frac) & blocked.any(dim=1)
            first = torch.where(blocked, cand_cost, torch.full_like(cand_cost, float("inf"))).argmin(dim=1)
            pick = torch.where(use, cand[rows, first], cand[:, 0])
            self._route_is_blocked[env_ids] = use | blocked[:, 0]
        else:
            pick = cost.argmin(dim=1)
            self._route_is_blocked[env_ids] = False
        goal = spots[rows, pick]
        self._goal_pos_w[env_ids] = goal
        self._goal_z_w[env_ids] = self._goal_spots_z[cell][rows, pick]

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

    def _ground_mesh(self):
        """The terrain as a Warp mesh in the world frame, read back off the
        stage on first use. The importer no longer keeps the mesh it made, and
        the RayCasters cast through Newton's scene with no mesh to share."""
        if self._ground is None:
            import numpy as np
            import isaaclab.sim as sim_utils
            from isaaclab.utils.warp import convert_to_warp_mesh
            from pxr import Usd, UsdGeom

            root = self.cfg.scene.terrain.prim_path
            prim = sim_utils.get_first_matching_child_prim(root, lambda p: p.GetTypeName() == "Mesh")
            if prim is None or not prim.IsValid():
                raise RuntimeError(f"no terrain mesh under {root} to set spawns down on")
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            tf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            points = points @ tf[:3, :3].T + tf[:3, 3]
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            self._ground = convert_to_warp_mesh(points, indices, device=self.device)
            print(f"[navigate] spawns ray-cast against {len(indices) // 3} terrain triangles from {prim.GetPath()}")
        return self._ground

    def _ground_heights(self, xy: torch.Tensor) -> torch.Tensor:
        """Terrain height under world xy, (..., 2) -> (...): NaN where the
        ray misses, which is off the terrain."""
        from isaaclab.utils.warp import raycast_mesh

        flat = xy.reshape(-1, 2)
        starts = torch.cat([flat, torch.full_like(flat[:, :1], RAY_TOP)], dim=-1)
        down = torch.zeros_like(starts)
        down[:, 2] = -1.0
        z = raycast_mesh(starts, down, self._ground_mesh())[0][:, 2]
        return torch.where(torch.isfinite(z), z, torch.full_like(z, float("nan"))).reshape(xy.shape[:-1])

    def _route_blocked(self, start: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        """(n, K): whether the machine's straight drive from start (n, 2) to
        each of ends (n, K, 2) meets a step in the ground taller than
        cfg.goal_blocked_step between neighbouring samples, read off the
        terrain mesh along ROUTE_LANES. A boulder's side and a crater's wall
        are steps of 0.35 m and more; the worked ground's pits and piles are
        far gentler."""
        n, k = ends.shape[:2]
        d = ends - start.unsqueeze(1)                                       # (n, K, 2)
        length = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1.0e-6)
        side = torch.stack([-d[..., 1], d[..., 0]], dim=-1) / length        # unit left of the route
        t = torch.linspace(0.0, 1.0, ROUTE_SAMPLES, device=self.device)
        lanes = torch.tensor(ROUTE_LANES, device=self.device)
        pts = (
            start[:, None, None, None, :]
            + d[:, :, None, None, :] * t[None, None, None, :, None]
            + side[:, :, None, None, :] * lanes[None, None, :, None, None]
        )                                                                   # (n, K, lanes, samples, 2)
        z = self._ground_heights(pts)
        step = torch.nan_to_num((z[..., 1:] - z[..., :-1]).abs(), nan=0.0)
        return (step > self.cfg.goal_blocked_step).flatten(2).any(dim=-1)

    def _draw_spawns(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A spawn for each env: world xy (n, 2), the ground height to set the
        tyres on (n,), and a yaw (n,).

        cfg.spawn_candidates spawn spots from the env's cell, each at a
        random heading. The ground is ray-cast under FOOTPRINT_XY at each, and
        the first candidate whose ground spans less than cfg.spawn_max_step is
        taken, or the most even one if none does. The height is the highest
        point under the tyres, so the highest tyre drops SPAWN_Z's 2 cm and
        none starts in the ground.
        """
        c = self.cfg
        n, k, dev = env_ids.numel(), max(int(c.spawn_candidates), 1), self.device
        cell = (self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids])
        spots = self._spawn_spots[cell]                                        # (n, P, 2)
        if c.spawn_radius is None:
            pick = torch.randint(0, spots.shape[1], (n, k), device=dev)
        else:
            # Spots within spawn_radius of the cell centre, or the nearest one
            # if none is.
            off = torch.linalg.norm(spots - self.scene.env_origins[env_ids, None, :2], dim=-1)
            near = off <= c.spawn_radius
            nearest = off <= off.amin(dim=1, keepdim=True)
            pick = torch.multinomial(torch.where(near.any(dim=1, keepdim=True), near, nearest).float(), k, replacement=True)
        xy = torch.gather(spots, 1, pick.unsqueeze(-1).expand(n, k, 2))        # (n, k, 2)
        yaw = (torch.rand(n, k, device=dev) * 2.0 - 1.0) * math.pi
        cos, sin = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
        fx, fy = self._footprint[:, 0], self._footprint[:, 1]
        pts = xy.unsqueeze(2) + torch.stack([cos * fx - sin * fy, sin * fx + cos * fy], dim=-1)
        z = self._ground_heights(pts)                                          # (n, k, F)
        span = torch.nan_to_num(z.amax(dim=-1) - z.amin(dim=-1), nan=float("inf"))
        ok = span < c.spawn_max_step
        best = torch.where(ok.any(dim=1), ok.float().argmax(dim=1), span.argmin(dim=1))
        rows = torch.arange(n, device=dev)
        ground = z[rows, best, : len(TYRE_XY)].amax(dim=-1)
        # Off the terrain under every candidate, which a spot inside its cell
        # cannot be: the cell's origin height rather than a NaN pose.
        ground = torch.where(torch.isfinite(ground), ground, self.scene.env_origins[env_ids, 2])
        return xy[rows, best], ground, yaw[rows, best]

    def _place_goals(self, env_ids: torch.Tensor, route) -> None:
        """cfg.fixed_route's goal, in each env's frame. The marker sits at
        the origin's height: the route's ground is level."""
        origin = self.scene.env_origins[env_ids]
        goal = torch.tensor([float(route[3]), float(route[4])], device=self.device)
        self._goal_pos_w[env_ids] = origin[:, :2] + goal
        self._goal_yaw[env_ids] = float(route[5])
        self._goal_z_w[env_ids] = origin[:, 2]

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Read before super() clears the episode buffers.
        exits = self._exit_stats(env_ids)
        exits["Episode/reached"] = float(self._arrived[env_ids].float().mean())
        exits["Episode/near"] = float(self._near[env_ids].float().mean())
        exits["Episode/failed"] = float(self._failed_now[env_ids].float().mean())
        exits["Episode/stuck"] = float(self._stuck[env_ids].float().mean())
        exits["Episode/blocked"] = float(self._route_is_blocked[env_ids].float().mean())
        exits["Episode/end_dist"] = float(self._prev_dist[env_ids].mean())
        exits["Episode/end_heading_deg"] = float(torch.rad2deg(self._last_herr[env_ids].abs()).mean())
        # Reads last episode's outcome, before it is cleared below.
        self._record_outcomes(env_ids)
        super()._reset_idx(env_ids)

        n = env_ids.numel()
        route = self.cfg.fixed_route
        if route is None:
            # Anywhere in the env's cell, facing anywhere; see _draw_spawns.
            xy, ground, yaw = self._draw_spawns(env_ids)
            origin = self.scene.env_origins[env_ids]
            spawn_xy = self._reset_robot(
                env_ids, yaw, self.cfg.arm_hold_angle, xy - origin[:, :2], z_offset=ground - origin[:, 2]
            )
            self._sample_goals(env_ids, spawn_xy)
        else:
            yaw = torch.full((n,), float(route[2]), device=self.device)
            xy_off = torch.tensor([float(route[0]), float(route[1])], device=self.device).expand(n, 2)
            spawn_xy = self._reset_robot(env_ids, yaw, self.cfg.arm_hold_angle, xy_off)
            self._place_goals(env_ids, route)
        # Initial distance and pose potential from the written spawn.
        d0 = torch.linalg.norm(self._goal_pos_w[env_ids] - spawn_xy, dim=-1)
        self._prev_dist[env_ids] = d0
        self._start_dist[env_ids] = d0.clamp_min(1e-3)
        self._prev_pose_pot[env_ids] = self._pose_potential(d0, R.axis_heading_error(self._goal_yaw[env_ids], yaw))
        self._last_herr[env_ids] = 0.0
        self._win_pos[env_ids] = spawn_xy.unsqueeze(1)
        self._win_yaw[env_ids] = yaw.unsqueeze(1)
        self._win_steps[env_ids] = 0
        self._stuck[env_ids] = False
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
        self.extras["log"]["Scan/mean_height"] = float(self._scan_mean)
        self.extras["log"]["Scan/at_clip"] = float(self._scan_at_clip)
