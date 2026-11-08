# Navigate: drive to a goal pose and hold it, following a planned path. The
# actor sees the path ahead and the goal, not the terrain. Actions are
# [forward, yaw]; the arms hold stowed, the drums still, the shroud inlets up.
#
# DirectRLEnv step order: _pre_physics_step, _apply_action (x decimation),
# _get_dones, _get_rewards, _reset_idx, _get_observations.

from __future__ import annotations

import math

import torch

from ..excavator import AXLE_X, TRACK, WHEEL_HALF_W, WHEEL_RADIUS, WHEEL_Y
from ..excavator_cfg import MAX_WHEEL_SPEED
from ..excavator_env_base import ExcavatorEnvBase
from ..mdp import paths as P
from ..mdp import rewards as R
from ..mdp.observations import NAV_NEAR_CELL, NAV_NEAR_IMAGE, navigate_obs_spec
from ..mdp.terrain import nav_near_pattern, scan_from_raycaster
from ..terrain_cfg import GOAL_PATCHES, SPAWN_PATCHES
from .navigate_env_cfg import NAV_CRITIC, ExcavatorNavigateEnvCfg

# Playback goal marker: a disc on the goal spot and a bar along its axis.
MARKER_RGB = (0.10, 0.90, 0.20)
MARKER_DISC = (0.5, 0.04)               # radius, height
MARKER_BAR = (5.0, 0.20, 0.04)          # length, width, height
MARKER_LIFT = 0.15                      # m above the goal spot
# Playback path: dots every PATH_DOT_SPACING m along the planned path.
PATH_RGB = (1.0, 0.55, 0.05)
PATH_DOT_RADIUS = 0.08
PATH_DOT_SPACING = 0.5
PATH_DOTS = 40

# Body-frame ground contact points: each tyre's contact line at its middle and
# both faces, then a grid across the tyres' outer faces and 1.25 m fore and aft.
TYRE_XY = tuple(
    (sx * AXLE_X, sy * WHEEL_Y + dy)
    for sx in (1, -1) for sy in (1, -1) for dy in (-WHEEL_HALF_W, 0.0, WHEEL_HALF_W)
)
_REACH_Y = WHEEL_Y + WHEEL_HALF_W
FOOTPRINT_XY = TYRE_XY + tuple((0.25 * i, 0.5 * j * _REACH_Y) for i in range(-5, 6) for j in range(-2, 3))
# Height the ground ray-casts start from, straight down.
RAY_TOP = 100.0
# Body-frame half-extents of the outline a collision is judged inside.
RAM_OUTLINE = (1.25, _REACH_Y)

# Blocked-route goals: candidates tested per reset, lateral offsets of the
# lines read along each route, and samples per line.
GOAL_CANDIDATES = 32
ROUTE_LANES = (-0.6, 0.0, 0.6)
ROUTE_SAMPLES = 73

REWARD_TERMS = ["progress", "bearing", "goal", "pose", "settle", "fail", "ram", "upright", "slip",
                "action_rate", "energy", "time"]


class ExcavatorNavigateEnv(ExcavatorEnvBase):
    cfg: ExcavatorNavigateEnvCfg

    def __init__(self, cfg: ExcavatorNavigateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        E, dev = self.num_envs, self.device

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=dev)

        self._actions = zeros(E, 2)
        self._prev_actions = zeros(E, 2)
        self._drum_fill = zeros(E, 2)               # no soil on this tier

        # Goal pose (world frame) and the ground height at the goal spot.
        self._goal_pos_w = zeros(E, 2)
        self._goal_yaw = zeros(E)
        self._goal_z_w = zeros(E)
        self._prev_dist = zeros(E)
        self._start_dist = zeros(E)
        self._prev_pose_pot = zeros(E)
        self._last_herr = zeros(E)

        # Arrival: _in_box inside the pose tolerances this step; _held steps in
        # a row inside them and stopped; _reached the hold completed this step;
        # _arrived it has completed this episode. _near is the curriculum's
        # success flag.
        self._in_box = zeros(E, dtype=torch.bool)
        self._held = zeros(E, dtype=torch.long)
        self._reached = zeros(E, dtype=torch.bool)
        self._arrived = zeros(E, dtype=torch.bool)
        self._near = zeros(E, dtype=torch.bool)
        self._failed_now = zeros(E, dtype=torch.bool)
        self._hold_left = zeros(E, dtype=torch.long)

        # Stuck check: ring buffers of the last stuck_window_steps positions and
        # headings.
        W = max(cfg.stuck_window_steps, 1)
        self._win_pos = zeros(E, W, 2)
        self._win_yaw = zeros(E, W)
        self._win_steps = zeros(E, dtype=torch.long)
        self._stuck = zeros(E, dtype=torch.bool)
        self._route_is_blocked = zeros(E, dtype=torch.bool)

        # Planned path: per-cell travel cost (built on the first reset), each
        # env's travel cost and cost-to-goal map, the cell centre they are laid
        # on, the map's value at the goal, last step's potential, and path over
        # straight distance at spawn.
        near_reach = 0.5 * (NAV_NEAR_IMAGE[2] - 1) * NAV_NEAR_CELL
        assert cfg.path_zone <= near_reach, f"path_zone {cfg.path_zone} m is beyond the near scan's {near_reach} m"
        self._path_axis = P.grid_axis(cfg.scene.terrain.terrain_generator.size[0], cfg.path_res, dev)
        G = self._path_axis.numel()
        self._path_cost = None
        self._path_env_cost = zeros(E, G, G)
        self._path = zeros(E, G, G)
        self._path_origin = zeros(E, 2)
        self._path_at_goal = zeros(E)
        self._prev_path = zeros(E)
        self._detour = torch.ones(E, device=dev)

        # Collision check: pairs of neighbouring near-scan cells inside
        # RAM_OUTLINE, along each grid axis.
        self._near_pattern = nav_near_pattern(dev)
        near = self._near_pattern.reshape(NAV_NEAR_IMAGE[1], NAV_NEAR_IMAGE[2], 2)
        inside = (near[..., 0].abs() <= RAM_OUTLINE[0]) & (near[..., 1].abs() <= RAM_OUTLINE[1])
        self._ram_pairs = (inside[1:, :] & inside[:-1, :], inside[:, 1:] & inside[:, :-1])
        self._rammed = zeros(E, dtype=torch.bool)

        self._ground = None                         # terrain as a Warp mesh, built on first use
        self._footprint = torch.tensor(FOOTPRINT_XY, device=dev)

        # Curriculum: goal range ceiling and a ring buffer of episode successes.
        self._goal_dist_hi = float(cfg.goal_dist_range[1])
        self._success_hist = zeros(cfg.curriculum_window)
        self._success_ptr = 0
        self._success_n = 0

        self._trace = None
        if cfg.goal_trace_envs > 0:
            self._trace_n = min(cfg.goal_trace_envs, E)
            self._trace_step = 0
            self._trace = open(cfg.goal_trace_path, "w", buffering=1)
            self._trace.write("step,ep_step,env,x,y,yaw,goal_x,goal_y,dist,herr,speed_fwd,speed,reached,timeout\n")

        self._goal_marker = self._make_goal_marker() if cfg.goal_marker else None
        self._log = R.TermLogger(REWARD_TERMS, E, dev)
        self._obs_spec = navigate_obs_spec()
        print(self._obs_spec.describe())
        print(NAV_CRITIC.describe())

    def _setup_scene(self):
        super()._setup_scene()
        self.far_scanner = self.scene["far_scanner"]
        self.near_scanner = self.scene["near_scanner"]
        self.terrain = self.scene.terrain
        # Flat-patch spots per cell, (rows, cols, spots, 3), world frame.
        patches = self.terrain.flat_patches
        self._goal_spots = patches[GOAL_PATCHES][..., :2]
        self._goal_spots_z = patches[GOAL_PATCHES][..., 2]
        self._spawn_spots = patches[SPAWN_PATCHES][..., :2]

    # --- actions ---

    def _pre_physics_step(self, actions: torch.Tensor):
        s = self.cfg.action_smoothing
        self._prev_actions[:] = self._actions
        self._actions[:] = (1.0 - s) * actions.clamp(-1.0, 1.0) + s * self._actions
        self._actions[self._arrived] = 0.0          # latched to zero once arrived

    def _apply_action(self):
        n = self.num_envs
        self._apply_drive(self._actions[:, 0], self._actions[:, 1])
        self._apply_arms(torch.full((n,), self.cfg.arm_hold_angle, device=self.device))
        self._apply_drums(torch.zeros(n, device=self.device))
        self._apply_shrouds(torch.full((n,), math.pi, device=self.device))

    # --- observations ---

    def _goal_in_body(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Body-frame goal vector (E, 2), distance (E,), (sin, cos) of twice the
        heading error (E, 2), and the heading error to the goal's axis (E,)."""
        yaw = self._base_yaw()
        vec_b = R.world_to_body_xy(self._goal_pos_w - self.robot.data.root_pos_w.torch[:, :2], yaw)
        heading = R.axis_heading_sin_cos(self._goal_yaw, yaw)
        return vec_b, torch.linalg.norm(vec_b, dim=-1), heading, 0.5 * torch.atan2(heading[:, 0], heading[:, 1])

    def _pose_potential(self, dist: torch.Tensor, herr: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return R.pose_potential(dist, herr, c.w_pos_fine, c.pos_sigma, c.w_heading, c.heading_sigma, c.heading_radius)

    def _terrain_scan(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean (far, near) scans: chassis z minus ground z under each ray."""
        chassis = self.robot.data.root_pos_w.torch
        return tuple(
            scan_from_raycaster(chassis, s.data.ray_hits_w.torch, clip=self.cfg.scan_clip)
            for s in (self.far_scanner, self.near_scanner)
        )

    def _path_points(self, distances: tuple[float, ...]) -> torch.Tensor:
        """(E, len(distances), 2) world xy of the points `distances` metres
        along the planned path from the chassis."""
        c = self.cfg
        corner = self._path_origin + self._path_axis[0]
        uv = P.trace_path(
            self._path,
            self._path_env_cost,
            (self.robot.data.root_pos_w.torch[:, :2] - corner) / c.path_res,
            (self._goal_pos_w - corner) / c.path_res,
            c.path_res,
            distances,
        )
        return uv * c.path_res + corner.unsqueeze(1)

    def _get_observations(self) -> dict:
        if self._goal_marker is not None:
            self._show_goal_marker()
        p = self._proprio()
        c = self.cfg
        vec_b, dist, heading, _ = self._goal_in_body()
        far, near = self._terrain_scan()
        ahead = self._path_points(c.path_lookahead) - self.robot.data.root_pos_w.torch[:, None, :2]
        k = ahead.shape[1]
        ahead_b = R.world_to_body_xy(ahead.reshape(-1, 2), self._base_yaw().repeat_interleave(k))
        policy = self._obs_spec.assemble({
            "base_lin_vel": p["base_lin_vel"],
            "base_ang_vel": p["base_ang_vel"],
            "projected_gravity": p["projected_gravity"],
            "path_points": ahead_b.reshape(self.num_envs, 2 * k),
            "goal_vec_b": vec_b / c.goal_dist_max,
            "goal_vec_near": vec_b / dist.clamp_min(c.goal_near_range).unsqueeze(-1),
            "goal_heading": heading,
            "wheel_vel": p["wheel_vel"],
            "last_action": self._actions,
        })
        wheel_vel = self.robot.data.joint_vel.torch
        v_fwd = p["base_lin_vel"][:, 0]
        critic = NAV_CRITIC.assemble({
            "terrain_scan_true": torch.cat([far, near], dim=-1),
            "drum_fill_mass": self._drum_fill,
            "base_lin_vel_w": self.robot.data.root_lin_vel_w.torch,
            "slip": torch.stack([
                R.wheel_slip(wheel_vel[:, self._left_ids], v_fwd, WHEEL_RADIUS),
                R.wheel_slip(wheel_vel[:, self._right_ids], v_fwd, WHEEL_RADIUS),
            ], dim=-1),
        })
        return {"policy": policy, "critic": critic}

    # --- rewards ---

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        p = self._proprio()
        d = self.robot.data
        _, dist, _, herr = self._goal_in_body()
        wheel_vel = d.joint_vel.torch[:, self._wheel_ids]
        torque = d.applied_torque.torch[:, self._wheel_ids]
        speed = torch.linalg.norm(p["base_lin_vel"][:, :2], dim=-1)
        pose_pot = self._pose_potential(dist, herr)
        pos = d.root_pos_w.torch[:, :2]
        path = self._path_potential(pos)
        way_b = R.world_to_body_xy(self._path_direction(pos), self._base_yaw())
        rammed_out = self._rammed & self._failed_now if c.ram_ends_episode else torch.zeros_like(self._rammed)

        reward = self._log.add_all({
            "progress": c.w_progress * R.progress_reward(self._prev_path, path),
            "bearing": c.w_bearing * R.fade_in(dist, *c.bearing_fade)
            * R.bearing_alignment(way_b, p["base_lin_vel"][:, 0], MAX_WHEEL_SPEED * WHEEL_RADIUS),
            "goal": c.w_goal * self._reached.float(),
            "pose": pose_pot - self._prev_pose_pot,
            "settle": -c.w_settle * self._in_box.float()
            * R.settle_penalty(speed, p["base_ang_vel"][:, 2], 0.5 * TRACK),
            "fail": -c.w_fail * (self._failed_now & ~rammed_out).float(),
            "ram": -c.w_ram * rammed_out.float(),
            "upright": -c.w_upright * R.upright_penalty(p["projected_gravity"]),
            "slip": -c.w_slip * R.wheel_slip(wheel_vel, p["base_lin_vel"][:, 0], WHEEL_RADIUS),
            "action_rate": -c.w_action_rate * R.action_rate_penalty(self._actions, self._prev_actions),
            "energy": -c.w_energy * R.energy_penalty(torque, wheel_vel),
            "time": -c.w_time * torch.ones_like(dist),
        })
        self._prev_dist[:] = dist
        self._prev_path[:] = path
        self._prev_pose_pot[:] = pose_pot
        return reward

    # --- terminations ---

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._failed_now = self._failed()
        _, dist, _, herr = self._goal_in_body()
        d = self.robot.data
        speed = torch.linalg.norm(d.root_lin_vel_b.torch[:, :2], dim=-1)
        yaw_rate = d.root_ang_vel_b.torch[:, 2]
        self._in_box = R.goal_pose_ok(dist, herr, c.goal_dist_tol, c.goal_heading_tol_rad)
        still = R.goal_reached(
            dist, herr, speed, yaw_rate, c.goal_dist_tol, c.goal_heading_tol_rad, c.goal_speed_tol, c.goal_yaw_rate_tol
        )
        self._held = torch.where(still, self._held + 1, torch.zeros_like(self._held))
        self._reached = (self._held >= c.arrive_hold_steps) & ~self._arrived
        self._arrived |= self._reached
        self._near |= (dist < c.curriculum_dist_tol) & (speed < c.curriculum_speed_tol)
        self._last_herr[:] = herr
        if c.stuck_window_steps > 0:
            self._stuck = self._stuck_now(dist)
            self._failed_now |= self._stuck
        self._rammed = self._rammed_now() & ~self._arrived
        if c.ram_ends_episode:
            self._failed_now |= self._rammed
        if self._trace is not None:
            self._write_trace(dist, herr, time_out)

        if c.goal_hold_steps == 0:
            return self._failed_now | self._reached, time_out
        # Run goal_hold_steps more steps after arriving, commands latched to zero.
        self._hold_left = torch.where(
            self._reached,
            torch.full_like(self._hold_left, c.goal_hold_steps),
            torch.where(self._arrived, self._hold_left - 1, self._hold_left),
        )
        return self._failed_now | (self._arrived & (self._hold_left <= 0)), time_out

    def _stuck_now(self, dist: torch.Tensor) -> torch.Tensor:
        """Beyond goal_near_range and not arrived, with net movement under
        stuck_min_move and net turn under stuck_min_turn across the last
        stuck_window_steps steps."""
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

    def _rammed_now(self) -> torch.Tensor:
        """A step over ram_step between neighbouring clean near-scan cells
        inside RAM_OUTLINE."""
        h = self._terrain_scan()[1].reshape(-1, NAV_NEAR_IMAGE[1], NAV_NEAR_IMAGE[2])
        along0, along1 = self._ram_pairs
        step0 = ((h[:, 1:, :] - h[:, :-1, :]).abs() > self.cfg.ram_step) & along0
        step1 = ((h[:, :, 1:] - h[:, :, :-1]).abs() > self.cfg.ram_step) & along1
        return step0.flatten(1).any(dim=1) | step1.flatten(1).any(dim=1)

    def _write_trace(self, dist: torch.Tensor, herr: torch.Tensor, time_out: torch.Tensor) -> None:
        """One CSV row per traced env per step, positions in the env frame,
        herr in degrees."""
        n = self._trace_n
        o = self.scene.env_origins[:n, :2]
        pos = (self.robot.data.root_pos_w.torch[:n, :2] - o).tolist()
        goal = (self._goal_pos_w[:n] - o).tolist()
        yaw = self._base_yaw()[:n].tolist()
        vel = self.robot.data.root_lin_vel_b.torch[:n, :2]
        spd = vel[:, 0].tolist()
        mag = torch.linalg.norm(vel, dim=-1).tolist()
        d = dist[:n].tolist()
        h = torch.rad2deg(herr[:n]).tolist()
        reached = self._arrived[:n].tolist()
        tout = time_out[:n].tolist()
        ep = self.episode_length_buf[:n].tolist()
        for i in range(n):
            self._trace.write(
                f"{self._trace_step},{ep[i]},{i},{pos[i][0]:.4f},{pos[i][1]:.4f},{yaw[i]:.4f},"
                f"{goal[i][0]:.4f},{goal[i][1]:.4f},{d[i]:.4f},{h[i]:.3f},{spd[i]:.4f},{mag[i]:.4f},"
                f"{int(reached[i])},{int(tout[i])}\n"
            )
        self._trace_step += 1

    # --- goal marker ---

    def _make_goal_marker(self):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        look = sim_utils.PreviewSurfaceCfg(diffuse_color=MARKER_RGB)
        return VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/navigate_goal",
            markers={
                "disc": sim_utils.CylinderCfg(radius=MARKER_DISC[0], height=MARKER_DISC[1], visual_material=look),
                "bar": sim_utils.CuboidCfg(size=MARKER_BAR, visual_material=look),
                "dot": sim_utils.SphereCfg(
                    radius=PATH_DOT_RADIUS, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=PATH_RGB)
                ),
            },
        ))

    def _show_goal_marker(self) -> None:
        """Per env, env-major: a disc and a bar at the goal, then PATH_DOTS dots
        along the planned path from the chassis."""
        n = self.num_envs
        at = torch.cat([self._goal_pos_w, (self._goal_z_w + MARKER_LIFT).unsqueeze(-1)], dim=-1)
        zero = torch.zeros_like(self._goal_yaw)
        quat = torch.stack([zero, zero, torch.sin(0.5 * self._goal_yaw), torch.cos(0.5 * self._goal_yaw)], dim=-1)
        dots = self._path_points(tuple(PATH_DOT_SPACING * (i + 1) for i in range(PATH_DOTS)))
        dot_z = torch.nan_to_num(self._ground_heights(dots), nan=0.0) + MARKER_LIFT
        dot_at = torch.cat([dots, dot_z.unsqueeze(-1)], dim=-1)                     # (n, PATH_DOTS, 3)
        dot_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(n, PATH_DOTS, 4)
        self._goal_marker.visualize(
            translations=torch.cat([at.unsqueeze(1).expand(n, 2, 3), dot_at], dim=1).reshape(-1, 3),
            orientations=torch.cat([quat.unsqueeze(1).expand(n, 2, 4), dot_quat], dim=1).reshape(-1, 4),
            marker_indices=torch.tensor([0, 1] + [2] * PATH_DOTS, device=self.device).repeat(n),
        )

    # --- reset ---

    def _record_outcomes(self, env_ids: torch.Tensor) -> None:
        """Move each env's terrain row and widen the goal range on the success
        rate over the last curriculum_window episodes."""
        c = self.cfg
        k = env_ids.numel()
        near = self._near[env_ids]
        stalled = ~near & (self._prev_dist[env_ids] > c.terrain_demote_fraction * self._start_dist[env_ids])
        self.terrain.update_env_origins(env_ids, near, stalled)

        idx = (self._success_ptr + torch.arange(k, device=self.device)) % c.curriculum_window
        self._success_hist[idx] = near.float()
        self._success_ptr = int((self._success_ptr + k) % c.curriculum_window)
        self._success_n = min(self._success_n + k, c.curriculum_window)
        if (
            self._success_n >= c.curriculum_window
            and float(self._success_hist.mean()) > c.curriculum_success_rate
            and self._goal_dist_hi < c.goal_dist_max
        ):
            self._goal_dist_hi = min(self._goal_dist_hi + c.curriculum_step, c.goal_dist_max)
            self._success_n = 0

    def _sample_goals(self, env_ids: torch.Tensor, spawn_xy: torch.Tensor) -> None:
        """Goal from the flat-patch spots of each env's cell.

        A distance is drawn from [goal_dist_range[0], goal_dist_hi]; the spot
        whose distance from the spawn is nearest it wins, spots nearer than the
        range's low end last. For goal_blocked_frac of envs the winner is
        instead the best of the GOAL_CANDIDATES nearest spots whose straight
        route is blocked, where one is.
        """
        c = self.cfg
        n, dev = env_ids.numel(), self.device
        rows = torch.arange(n, device=dev)
        cell = (self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids])
        spots = self._goal_spots[cell]                                      # (n, P, 2)
        reach = torch.linalg.norm(spots - spawn_xy.unsqueeze(1), dim=-1)   # (n, P)
        lo = c.goal_dist_range[0]
        want = lo + (self._goal_dist_hi - lo) * torch.rand(n, 1, device=dev)
        cost = (reach - want).abs() + (reach < lo).float() * 1.0e3

        cand_cost, cand = cost.topk(GOAL_CANDIDATES, dim=1, largest=False)
        ends = torch.gather(spots, 1, cand.unsqueeze(-1).expand(n, GOAL_CANDIDATES, 2))
        blocked = self._route_blocked(spawn_xy, ends) & (cand_cost < 1.0e3)
        use = (torch.rand(n, device=dev) < c.goal_blocked_frac) & blocked.any(dim=1)
        first = torch.where(blocked, cand_cost, torch.full_like(cand_cost, float("inf"))).argmin(dim=1)
        pick = torch.where(use, cand[rows, first], cand[:, 0])
        self._route_is_blocked[env_ids] = use | blocked[:, 0]

        goal = spots[rows, pick]
        self._goal_pos_w[env_ids] = goal
        self._goal_z_w[env_ids] = self._goal_spots_z[cell][rows, pick]
        rel = goal - spawn_xy
        yaw = torch.atan2(rel[:, 1], rel[:, 0]) + torch.randn(n, device=dev) * c.goal_yaw_spread
        free = torch.rand(n, device=dev) < c.goal_yaw_random_frac
        self._goal_yaw[env_ids] = torch.where(free, (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi, yaw)

    def _ground_mesh(self):
        """The terrain mesh from the stage as a world-frame Warp mesh."""
        if self._ground is None:
            import numpy as np
            import isaaclab.sim as sim_utils
            from isaaclab.utils.warp import convert_to_warp_mesh
            from pxr import Usd, UsdGeom

            prim = sim_utils.get_first_matching_child_prim(
                self.cfg.scene.terrain.prim_path, lambda p: p.GetTypeName() == "Mesh"
            )
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            tf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            points = points @ tf[:3, :3].T + tf[:3, 3]
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            self._ground = convert_to_warp_mesh(points, indices, device=self.device)
        return self._ground

    def _ground_heights(self, xy: torch.Tensor) -> torch.Tensor:
        """Terrain height under world xy (..., 2) -> (...); NaN off the terrain."""
        from isaaclab.utils.warp import raycast_mesh

        flat = xy.reshape(-1, 2)
        starts = torch.cat([flat, torch.full_like(flat[:, :1], RAY_TOP)], dim=-1)
        down = torch.zeros_like(starts)
        down[:, 2] = -1.0
        z = raycast_mesh(starts, down, self._ground_mesh())[0][:, 2]
        return torch.where(torch.isfinite(z), z, torch.full_like(z, float("nan"))).reshape(xy.shape[:-1])

    def _route_blocked(self, start: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        """(n, K): the straight drive from start (n, 2) to each of ends
        (n, K, 2) steps by more than goal_blocked_step between neighbouring
        samples along any of ROUTE_LANES."""
        d = ends - start.unsqueeze(1)
        side = torch.stack([-d[..., 1], d[..., 0]], dim=-1) / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1.0e-6)
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

    # --- path progress ---

    def _cell_path_cost(self) -> torch.Tensor:
        """(rows, cols, G, G) travel cost of every terrain cell, built on first use."""
        if self._path_cost is None:
            c = self.cfg
            origins = self.terrain.terrain_origins
            gx, gy = torch.meshgrid(self._path_axis, self._path_axis, indexing="ij")
            local = torch.stack([gx, gy], dim=-1)
            self._path_cost = torch.stack([
                P.obstacle_cost(
                    self._ground_heights(origins[r, :, None, None, :2] + local),
                    c.path_res, c.goal_blocked_step, c.path_clearance, c.path_zone, c.path_zone_cost,
                    c.path_obstacle_cost,
                )
                for r in range(origins.shape[0])
            ])
        return self._path_cost

    def _build_paths(self, env_ids: torch.Tensor) -> None:
        """Each env's cost-to-goal map over its cell, from its current goal."""
        cell = (self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids])
        origin = self.terrain.terrain_origins[cell][:, :2]
        goal_uv = (self._goal_pos_w[env_ids] - origin - self._path_axis[0]) / self.cfg.path_res
        cost = self._cell_path_cost()[cell]
        field = P.path_field(cost, goal_uv, self.cfg.path_res, self.cfg.path_max_iters)
        self._path_env_cost[env_ids] = cost
        self._path[env_ids] = field
        self._path_origin[env_ids] = origin
        self._path_at_goal[env_ids] = P.sample_field(field, goal_uv, self.cfg.path_res)

    def _path_potential(self, xy_w: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Path cost to the goal from world xy, 0 at the goal."""
        ids = slice(None) if env_ids is None else env_ids
        uv = (xy_w - self._path_origin[ids] - self._path_axis[0]) / self.cfg.path_res
        return P.sample_field(self._path[ids], uv, self.cfg.path_res) - self._path_at_goal[ids]

    def _path_direction(self, xy_w: torch.Tensor) -> torch.Tensor:
        """(E, 2) world unit vector along which the path cost falls fastest at
        xy; straight at the goal where the map is flat."""
        h = self.cfg.path_res
        ex = torch.tensor([h, 0.0], device=self.device)
        ey = torch.tensor([0.0, h], device=self.device)
        grad = torch.stack([
            self._path_potential(xy_w + ex) - self._path_potential(xy_w - ex),
            self._path_potential(xy_w + ey) - self._path_potential(xy_w - ey),
        ], dim=-1)
        norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
        straight = self._goal_pos_w - xy_w
        straight = straight / torch.linalg.norm(straight, dim=-1, keepdim=True).clamp_min(1.0e-6)
        return torch.where(norm > 1.0e-6, -grad / norm.clamp_min(1.0e-6), straight)

    def _draw_spawns(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Spawn xy (n, 2), ground height under the tyres (n,) and yaw (n,).

        spawn_candidates spawn spots of the env's cell at random headings; the
        first whose ground under FOOTPRINT_XY spans less than spawn_max_step,
        else the most even. The height is the highest point under the tyres.
        """
        c = self.cfg
        n, k, dev = env_ids.numel(), c.spawn_candidates, self.device
        cell = (self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids])
        spots = self._spawn_spots[cell]                                        # (n, P, 2)
        if c.spawn_radius is None:
            pick = torch.randint(0, spots.shape[1], (n, k), device=dev)
        else:
            # Spots within spawn_radius of the cell centre, else the nearest.
            off = torch.linalg.norm(spots - self.scene.env_origins[env_ids, None, :2], dim=-1)
            near = off <= c.spawn_radius
            nearest = off <= off.amin(dim=1, keepdim=True)
            pick = torch.multinomial(torch.where(near.any(dim=1, keepdim=True), near, nearest).float(), k, replacement=True)
        xy = torch.gather(spots, 1, pick.unsqueeze(-1).expand(n, k, 2))        # (n, k, 2)
        yaw = (torch.rand(n, k, device=dev) * 2.0 - 1.0) * math.pi
        cos, sin = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
        fx, fy = self._footprint[:, 0], self._footprint[:, 1]
        z = self._ground_heights(xy.unsqueeze(2) + torch.stack([cos * fx - sin * fy, sin * fx + cos * fy], dim=-1))
        span = torch.nan_to_num(z.amax(dim=-1) - z.amin(dim=-1), nan=float("inf"))
        ok = span < c.spawn_max_step
        best = torch.where(ok.any(dim=1), ok.float().argmax(dim=1), span.argmin(dim=1))
        rows = torch.arange(n, device=dev)
        return xy[rows, best], z[rows, best, : len(TYRE_XY)].amax(dim=-1), yaw[rows, best]

    def _place_goals(self, env_ids: torch.Tensor, route) -> None:
        """cfg.fixed_route's goal in each env's frame."""
        origin = self.scene.env_origins[env_ids]
        self._goal_pos_w[env_ids] = origin[:, :2] + torch.tensor([route[3], route[4]], device=self.device)
        self._goal_yaw[env_ids] = route[5]
        self._goal_z_w[env_ids] = origin[:, 2]

    def _reset_idx(self, env_ids: torch.Tensor):
        exits = self._exit_stats(env_ids)
        exits.update({
            "Episode/reached": float(self._arrived[env_ids].float().mean()),
            "Episode/near": float(self._near[env_ids].float().mean()),
            "Episode/failed": float(self._failed_now[env_ids].float().mean()),
            "Episode/stuck": float(self._stuck[env_ids].float().mean()),
            "Episode/rammed": float(self._rammed[env_ids].float().mean()),
            "Episode/blocked": float(self._route_is_blocked[env_ids].float().mean()),
            "Episode/detour": float(self._detour[env_ids].mean()),
            "Episode/end_dist": float(self._prev_dist[env_ids].mean()),
            "Episode/end_heading_deg": float(torch.rad2deg(self._last_herr[env_ids].abs()).mean()),
        })
        self._record_outcomes(env_ids)
        super()._reset_idx(env_ids)

        n = env_ids.numel()
        route = self.cfg.fixed_route
        if route is None:
            xy, ground, yaw = self._draw_spawns(env_ids)
            origin = self.scene.env_origins[env_ids]
            spawn_xy = self._reset_robot(
                env_ids, yaw, self.cfg.arm_hold_angle, xy - origin[:, :2], z_offset=ground - origin[:, 2]
            )
            self._sample_goals(env_ids, spawn_xy)
        else:
            yaw = torch.full((n,), route[2], device=self.device)
            xy_off = torch.tensor([route[0], route[1]], device=self.device).expand(n, 2)
            spawn_xy = self._reset_robot(env_ids, yaw, self.cfg.arm_hold_angle, xy_off)
            self._place_goals(env_ids, route)

        d0 = torch.linalg.norm(self._goal_pos_w[env_ids] - spawn_xy, dim=-1)
        self._build_paths(env_ids)
        path0 = self._path_potential(spawn_xy, env_ids)
        self._prev_path[env_ids] = path0
        self._detour[env_ids] = path0 / d0.clamp_min(1.0e-3)
        self._prev_dist[env_ids] = d0
        self._start_dist[env_ids] = d0.clamp_min(1e-3)
        self._prev_pose_pot[env_ids] = self._pose_potential(d0, R.axis_heading_error(self._goal_yaw[env_ids], yaw))
        self._win_pos[env_ids] = spawn_xy.unsqueeze(1)
        self._win_yaw[env_ids] = yaw.unsqueeze(1)
        for buf in (self._last_herr, self._win_steps, self._held, self._hold_left, self._actions, self._prev_actions):
            buf[env_ids] = 0
        for flag in (self._stuck, self._rammed, self._in_box, self._reached, self._arrived, self._near, self._failed_now):
            flag[env_ids] = False

        self.extras["log"] = self._log.flush(env_ids)
        self.extras["log"].update(exits)
        self.extras["log"]["Curriculum/goal_dist_hi"] = self._goal_dist_hi
        self.extras["log"]["Curriculum/terrain_level"] = float(self.terrain.terrain_levels.float().mean())
