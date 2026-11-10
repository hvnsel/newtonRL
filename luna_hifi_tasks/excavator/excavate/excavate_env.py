# excavate_env.py
#
# Cut a strip of an MPM regolith bed in one straight pass and carry the spoil
# in the drums. The env runs the tool: after the settle, over dig_ramp_seconds
# with the wheels held, both arms lower to the dig angle, both shrouds turn to
# hold the inlet at dig_inlet_tilt off straight down, and both drums spin up.
# Then the policy drives: actions are [forward, yaw] in [-1, 1]. The episode
# ends when the front drum passes the end of the work area.

from __future__ import annotations

import math

import newton
import numpy as np
import torch
import warp as wp

from isaaclab_newton.assets import MPMObject

from ..excavator import ARM_LEN, PIVOT_X, ROTOR_VANES, WHEEL_RADIUS
from ..excavator_cfg import MAX_YAW_SPEED, usd_status
from ..excavator_env_base import ExcavatorEnvBase
from ..mdp import rewards as R
from ..mdp.sensors import drum_fill_mass, mpm_grid_particle_mass, mpm_particle_state, soil_heightmap
from ..mdp.terrain import (
    degrade_scan,
    dig_critic_far_pattern,
    dig_critic_near_pattern,
    dig_scan_pattern,
    sample_height_grid,
    scan_from_heightfield,
    scan_points_world,
)
from .excavate_env_cfg import (
    BED_FLOOR_Z,
    BORE_CHORD_Z,
    BORE_HALF_LEN,
    BORE_RADIUS,
    DIG_CRITIC,
    DIG_OBS,
    FOOTPRINT_HALF_X,
    FOOTPRINT_HALF_Y,
    ExcavatorExcavateEnvCfg,
)


class ExcavatorExcavateEnv(ExcavatorEnvBase):
    cfg: ExcavatorExcavateEnvCfg

    def __init__(self, cfg: ExcavatorExcavateEnvCfg, render_mode: str | None = None, **kwargs):
        problem = usd_status()
        if problem is not None:
            raise RuntimeError(f"[asset] {problem}")

        # Re-derive the bed, grid capacities and MPM material for this env count.
        n = int(cfg.scene.num_envs)
        assert n <= cfg.max_num_envs, (
            f"num_envs={n} exceeds max_num_envs={cfg.max_num_envs}; raise max_num_envs in the cfg"
        )
        cfg.max_num_envs = n
        cfg.__post_init__()
        super().__init__(cfg, render_mode, **kwargs)

        E, dev = self.num_envs, self.device
        self._actions = torch.zeros(E, 2, device=dev)
        self._prev_actions = torch.zeros(E, 2, device=dev)

        self._particle_mass = mpm_grid_particle_mass(cfg.scene.soil.spawn)
        self._fill_kg = torch.zeros(E, 2, device=dev)           # front, rear

        self._dig_pattern = dig_scan_pattern(dev)
        self._far_pattern = dig_critic_far_pattern(dev)
        self._near_pattern = dig_critic_near_pattern(dev)
        # Scan cells over the lead drum, in its frame.
        self._footprint = (
            (self._dig_pattern[:, 0].abs() <= FOOTPRINT_HALF_X)
            & (self._dig_pattern[:, 1].abs() <= FOOTPRINT_HALF_Y)
        )

        # (ny, nx, 2) env-local xy of every bed-grid cell centre.
        gx = cfg.bed_grid_lower[0] + (torch.arange(cfg.bed_grid_nx, device=dev) + 0.5) * cfg.voxel_size
        gy = cfg.bed_grid_lower[1] + (torch.arange(cfg.bed_grid_ny, device=dev) + 0.5) * cfg.voxel_size
        self._grid_xy = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1)
        self._grid_cell_area = cfg.voxel_size ** 2
        # Bed-grid cells this episode's cut covers, and the fraction at or below the plane.
        self._work_mask = torch.zeros(E, cfg.bed_grid_ny, cfg.bed_grid_nx, dtype=torch.bool, device=dev)
        self._work_done = torch.zeros(E, device=dev)

        # Commanded cut plane, world frame: z(X, Y) = z0 + grad . ([X, Y] - xy).
        self._target_xy = torch.zeros(E, 2, device=dev)
        self._target_z0 = torch.zeros(E, device=dev)
        self._target_grad = torch.zeros(E, 2, device=dev)
        self._target_depth = torch.zeros(E, device=dev)         # m below the settled surface

        self._shape_done = torch.zeros(E, dtype=torch.bool, device=dev)
        self._shape_newly_done = torch.zeros(E, dtype=torch.bool, device=dev)
        self._drum_full = torch.zeros(E, dtype=torch.bool, device=dev)
        self._failed_now = torch.zeros(E, dtype=torch.bool, device=dev)
        self._nonfinite_now = torch.zeros(E, dtype=torch.bool, device=dev)
        self._nonfinite_last: tuple[int, ...] = ()
        self._nonfinite_since = 0
        self._off_bed = torch.zeros(E, dtype=torch.bool, device=dev)
        self._escaped = torch.zeros(E, dtype=torch.bool, device=dev)
        # Furthest any particle has been from its env origin this episode, m.
        self._particle_max_m = torch.zeros(E, device=dev)
        # Signed weighted reward terms of the last step.
        self._last_terms: dict[str, torch.Tensor] = {}
        # (E, ny, nx) env-local bed heights from the last reward step.
        self._bed_grid = torch.full((E, cfg.bed_grid_ny, cfg.bed_grid_nx), cfg.bed_top, device=dev)
        # Episodes since each env's soil was re-laid, staggered across envs.
        self._soil_age = torch.randint(0, max(cfg.soil_reset_every, 1), (E,), device=dev)
        # Work-area soil volume above the plane when the settle ended, m3.
        self._above_init = torch.zeros(E, device=dev)
        # Previous step's removal, unheld and below-plane volume, for the deltas.
        self._removal_prev = torch.zeros(E, device=dev)
        self._unheld_prev = torch.zeros(E, device=dev)
        self._below_prev = torch.zeros(E, device=dev)
        # Mass balance in kg, for the log.
        self._removed_kg = torch.zeros(E, device=dev)
        self._cut_total_kg = torch.zeros(E, device=dev)
        self._payload_kg = torch.zeros(E, device=dev)
        # Leading drum, 0 front and 1 rear.
        self._lead = torch.zeros(E, dtype=torch.long, device=dev)

        # Settle steps left per env; _cut_fresh is True while any remain.
        self._cut_fresh = torch.ones(E, dtype=torch.bool, device=dev)
        step_dt = cfg.sim.dt * cfg.decimation
        self._settle_steps = max(1, round(cfg.settle_seconds / step_dt))
        self._settle_left = torch.full((E,), self._settle_steps, dtype=torch.long, device=dev)

        # Tool: steps since the settle ended (the ramp runs over the first
        # _ramp_steps of them), the arm angle each env spawned with, the dig
        # arm and shroud angles, the handover heading, and whether the front
        # drum has passed the end of the work area.
        self._ramp_steps = max(1, round(cfg.dig_ramp_seconds / step_dt))
        self._since_settle = torch.zeros(E, dtype=torch.long, device=dev)
        self._arm_start = torch.zeros(E, device=dev)
        _, self._dig_arm = cfg.boom_command_for_cut(cfg.dig_cut)
        self._dig_shroud = -self._dig_arm + cfg.dig_inlet_tilt
        self._work_yaw = torch.zeros(E, device=dev)
        self._pass_done = torch.zeros(E, dtype=torch.bool, device=dev)

        # Load sensor: first-order lag and a per-episode calibration bias.
        self._fill_filt = torch.zeros(E, 2, device=dev)
        self._fill_bias = torch.zeros(E, 2, device=dev)
        self._fill_alpha = step_dt / (cfg.fill_sensor_tau + step_dt)
        # Slower lag that the spill term reads.
        self._spill_filt = torch.zeros(E, 2, device=dev)
        self._spill_filt_prev = torch.zeros(E, 2, device=dev)
        self._spill_alpha = step_dt / (cfg.spill_filter_tau + step_dt)

        self._log = R.TermLogger(
            ["removal", "unheld", "success", "fail", "overcut", "spill",
             "stall", "drift", "upright", "energy", "action_rate", "time"],
            E, dev,
        )

        print(DIG_OBS.describe())
        print(DIG_CRITIC.describe())
        mat = cfg.scene.soil.spawn.material
        print(
            f"[excavate] particle mass {self._particle_mass:.4f} kg, "
            f"target load {cfg.target_load_kg:.1f} kg "
            f"(the drum interior holds {cfg.drum_capacity_kg:.1f}), "
            f"cut ref {cfg.cut_volume_ref:.4f} m3, "
            f"bed grid {cfg.bed_grid_nx} x {cfg.bed_grid_ny} @ {cfg.voxel_size:.3f} m"
        )
        print(
            f"[excavate] soil as spawned: density {mat.density:.0f} kg/m3, "
            f"friction {mat.friction:.2f}, cohesion (yield_stress) {mat.yield_stress:.0f} Pa, "
            f"yield_pressure {mat.yield_pressure:.3g} Pa"
        )
        print(
            f"[excavate] tool: arm {self._dig_arm:+.3f} rad, shroud {self._dig_shroud:+.3f} rad, "
            f"drums {cfg.dig_drum_speed:+.2f} rad/s"
            + ("" if cfg.dig_tool else " (dig_tool False: arms held at the start angle)")
        )

    def _setup_scene(self):
        super()._setup_scene()
        self.soil: MPMObject = self.scene["soil"]

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        s = self.cfg.action_smoothing
        self._prev_actions[:] = self._actions
        self._actions[:] = (1.0 - s) * actions.clamp(-1.0, 1.0) + s * self._actions

    def _apply_action(self):
        c = self.cfg
        a = self._actions
        driving = self._since_settle >= self._ramp_steps
        ramp = (self._since_settle.float() / self._ramp_steps).clamp(0.0, 1.0) * float(c.dig_tool)
        # Forward: dig_forward * (1 + a0) of top speed. Yaw rate: a heading hold
        # on the handover heading plus dig_yaw_rate * a1. Wheels held until the
        # ramp is done.
        err = self._work_yaw - self._base_yaw()
        err = torch.atan2(torch.sin(err), torch.cos(err))
        hold = (c.heading_hold_gain * err).clamp(-c.dig_yaw_rate, c.dig_yaw_rate)
        forward = torch.where(driving, c.dig_forward * (1.0 + a[:, 0]), torch.zeros_like(err))
        yaw_rate = torch.where(driving, hold + c.dig_yaw_rate * a[:, 1], torch.zeros_like(err))
        self._apply_drive(forward, yaw_rate / MAX_YAW_SPEED)
        self._update_lead()
        self._apply_arms(self._arm_start + ramp * (self._dig_arm - self._arm_start))
        self._apply_drums(ramp * c.dig_drum_speed)
        self._apply_shrouds(ramp * self._dig_shroud)

    # ------------------------------------------------------------------
    # soil sensing
    # ------------------------------------------------------------------

    def _update_lead(self) -> None:
        """Lead drum from the sign of the commanded drive; held inside +-dig_lead_deadband."""
        v = self._forward_cmd
        db = self.cfg.dig_lead_deadband
        front = torch.zeros_like(self._lead)
        rear = torch.ones_like(self._lead)
        self._lead = torch.where(v > db, front, torch.where(v < -db, rear, self._lead))

    def _lead_drum(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(E, 3) position of the leading drum and (E,) yaw offset of its scan:
        0 for the front drum, pi for the rear."""
        pos = self._drum_poses()[0]
        idx = self._lead.view(-1, 1, 1).expand(-1, 1, pos.shape[-1])
        return pos.gather(1, idx).squeeze(1), self._lead.to(pos.dtype) * math.pi

    def _particles(self) -> tuple[torch.Tensor, torch.Tensor]:
        return mpm_particle_state(self.soil)

    def _compute_fill(self, pos: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """(E, 2) kg of soil inside the front and rear drums' shells."""
        spos, squat = self._shroud_poses()
        out = torch.zeros(self.num_envs, 2, device=self.device)
        for i in range(2):
            out[:, i] = drum_fill_mass(
                pos, env, self._particle_mass, spos[:, i], squat[:, i],
                BORE_RADIUS, BORE_HALF_LEN, self.num_envs, BORE_CHORD_Z,
            )
        return out

    def _bed_heightmap(self, pos: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return soil_heightmap(
            pos, env, self.scene.env_origins,
            c.bed_grid_lower, c.voxel_size, c.bed_grid_nx, c.bed_grid_ny, BED_FLOOR_Z,
        )

    def _scans(self, bed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(actor scan at the lead drum, critic scan around the chassis) from one bed grid."""
        c = self.cfg
        d = self.robot.data
        quat = d.root_quat_w.torch
        chassis = d.root_pos_w.torch
        lead, lead_yaw = self._lead_drum()

        def sample(origin, pattern, yaw_offset=None):
            return scan_from_heightfield(
                bed, origin, quat, pattern, self.scene.env_origins,
                c.bed_grid_lower, c.voxel_size, origin[:, 2], clip=c.scan_clip,
                yaw_offset=yaw_offset,
            )

        actor = sample(lead, self._dig_pattern, lead_yaw)
        critic = torch.cat(
            [sample(chassis, self._far_pattern), sample(chassis, self._near_pattern)], dim=-1
        )
        return actor, critic

    def _plane_z(self, xy: torch.Tensor) -> torch.Tensor:
        """z of the commanded plane at xy (E, ..., 2), clamped to the bed floor. (E, ...)."""
        nd = xy.dim() - 2
        rel = xy - self._target_xy.view(-1, *([1] * nd), 2)
        z = self._target_z0.view(-1, *([1] * nd)) + (
            rel * self._target_grad.view(-1, *([1] * nd), 2)
        ).sum(dim=-1)
        floor = self.scene.env_origins[:, 2] + BED_FLOOR_Z
        return torch.maximum(z, floor.view(-1, *([1] * nd)))

    def _cut_state(self, actor_scan: torch.Tensor) -> dict[str, torch.Tensor]:
        """The cut command in the scan's frame, and how far the ground is from it.

        Heights are relative to the lead drum and clipped to +-scan_clip, the
        convention of the terrain scan: reference minus ground, so lower ground
        reads larger. The volumes are over the work area, in m3.
        """
        clip = self.cfg.scan_clip
        front, lead_yaw = self._lead_drum()
        drum_z = front[:, 2]

        pts = scan_points_world(
            front, self.robot.data.root_quat_w.torch, self._dig_pattern, lead_yaw
        )[:, self._footprint]
        t = (drum_z.unsqueeze(-1) - self._plane_z(pts)).clamp(-clip, clip)
        level = (drum_z - self._plane_z(front[:, :2])).clamp(-clip, clip)
        grad_b = R.world_to_body_xy(self._target_grad, self._base_yaw() + lead_yaw)
        # ground - target under the drum footprint; positive is soil above the plane.
        residual = t - actor_scan[:, self._footprint]

        bed_z = self._bed_grid + self.scene.env_origins[:, 2].view(-1, 1, 1)
        plane_z = self._plane_z(self._grid_xy.unsqueeze(0))
        # Cells at the floor hold no soil and are left out of both volumes.
        floor = (self.scene.env_origins[:, 2] + BED_FLOOR_Z).view(-1, 1, 1)
        soil = self._work_mask & (bed_z > floor + 0.5 * self.cfg.voxel_size)
        cell_residual = torch.where(soil, bed_z - plane_z, torch.zeros_like(bed_z))
        area = self._grid_cell_area
        cells = self._work_mask.flatten(1).sum(-1).clamp_min(1)
        done = (self._work_mask & (bed_z <= plane_z)).flatten(1).sum(-1)
        self._work_done = (done / cells).float()
        return {
            "target_level": level.unsqueeze(-1),
            "grad_forward": grad_b[:, 0:1],
            "grad_lateral": grad_b[:, 1:2],
            "depth_error": residual.mean(dim=-1, keepdim=True),
            "cut_progress": self._work_done.unsqueeze(-1),
            "above_volume": cell_residual.clamp_min(0.0).flatten(1).sum(-1) * area,
            "below_volume": (-cell_residual).clamp_min(0.0).flatten(1).sum(-1) * area,
        }

    def _drum_spawn_ground(
        self,
        env_ids: torch.Tensor,
        spawn_xy: torch.Tensor,
        yaw: torch.Tensor,
        arm_angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(world xy, world z) of the last step's bed surface under the front
        drum at spawn. The xy anchors the commanded plane."""
        c = self.cfg
        reach = PIVOT_X + ARM_LEN * torch.cos(arm_angle)
        xy = torch.stack([
            spawn_xy[:, 0] + reach * torch.cos(yaw),
            spawn_xy[:, 1] + reach * torch.sin(yaw),
        ], dim=-1)
        h = sample_height_grid(
            self._bed_grid[env_ids], xy.unsqueeze(1), self.scene.env_origins[env_ids],
            c.bed_grid_lower, c.voxel_size, c.bed_top,
        )[:, 0]
        return xy, self.scene.env_origins[env_ids, 2] + h

    def _set_work_area(
        self,
        env_ids: torch.Tensor,
        anchor_xy: torch.Tensor,
        yaw: torch.Tensor,
    ) -> None:
        """Mark the work area: a rectangle at the anchor, aligned with `yaw`."""
        c = self.cfg
        rel = self._grid_xy.unsqueeze(0) - (
            anchor_xy - self.scene.env_origins[env_ids, :2]
        ).view(-1, 1, 1, 2)
        cos, sin = torch.cos(yaw).view(-1, 1, 1), torch.sin(yaw).view(-1, 1, 1)
        fwd = rel[..., 0] * cos + rel[..., 1] * sin
        lat = -rel[..., 0] * sin + rel[..., 1] * cos
        self._work_mask[env_ids] = (
            (fwd >= -c.work_area_behind)
            & (fwd <= c.work_area_length)
            & (lat.abs() <= 0.5 * c.work_area_width)
        )

    def _fill_observed(self) -> torch.Tensor:
        """(E, 2) sensed load as a fraction of target_load_kg."""
        c = self.cfg
        reading = self._fill_filt * (1.0 + self._fill_bias)
        noise = torch.randn_like(reading) * (c.fill_sensor_noise * reading.abs() + c.fill_sensor_abs_kg)
        return ((reading + noise) / c.target_load_kg).clamp(-0.5, 3.0)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        c = self.cfg
        p = self._proprio()
        pos, env = self._particles()
        actor_scan, critic_scan = self._scans(self._bed_heightmap(pos, env))
        cut = self._cut_state(actor_scan)
        policy = DIG_OBS.assemble({
            "base_lin_vel": p["base_lin_vel"],
            "base_ang_vel": p["base_ang_vel"],
            "projected_gravity": p["projected_gravity"],
            "wheel_vel": p["wheel_vel"],
            "arm_pos": p["arm_pos"],
            "arm_vel": p["arm_vel"],
            "drum_vel": p["drum_vel"],
            "shroud_pos": p["shroud_pos"],
            "drum_phase": p["drum_phase"],
            "arm_torque": p["arm_torque"],
            "drum_torque": p["drum_torque"],
            "drum_fill": self._fill_observed(),
            "target_level": cut["target_level"],
            "grad_forward": cut["grad_forward"],
            "grad_lateral": cut["grad_lateral"],
            "depth_error": cut["depth_error"],
            "cut_progress": cut["cut_progress"],
            "terrain_scan": degrade_scan(
                actor_scan, self._dig_pattern, c.scan_noise_std, c.scan_dropout,
                c.scan_range_ref, c.scan_invalid,
            ),
            "last_action": self._actions,
        })

        wheel_vel = self.robot.data.joint_vel.torch
        v_fwd = p["base_lin_vel"][:, 0]
        slip = torch.stack([
            R.wheel_slip(wheel_vel[:, self._left_ids], v_fwd, WHEEL_RADIUS),
            R.wheel_slip(wheel_vel[:, self._right_ids], v_fwd, WHEEL_RADIUS),
        ], dim=-1)
        critic = DIG_CRITIC.assemble({
            "terrain_scan_true": critic_scan,
            "drum_fill_mass": self._fill_kg,
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
        d = self.robot.data

        pos, env = self._particles()
        d_env = torch.linalg.norm(pos - self.scene.env_origins[env], dim=-1)
        d_env = torch.nan_to_num(d_env, nan=0.0, posinf=0.0, neginf=0.0)
        self._particle_max_m.scatter_reduce_(0, env, d_env, reduce="amax")
        self._bed_grid = self._bed_heightmap(pos, env)

        # During the settle the plane is re-placed target_depth under the mean
        # soil surface of the work area; it holds there afterwards.
        work = self._work_mask & (self._bed_grid > BED_FLOOR_Z + 0.5 * self.cfg.voxel_size)
        cells = work.sum(dim=(1, 2))
        surface = (
            torch.where(work, self._bed_grid, torch.zeros_like(self._bed_grid)).sum(dim=(1, 2))
            / cells.clamp_min(1)
            + self.scene.env_origins[:, 2]
        )
        floor = self.scene.env_origins[:, 2] + BED_FLOOR_Z
        self._target_z0.copy_(torch.where(
            self._cut_fresh & (cells > 0),
            (surface - self._target_depth).clamp_min(floor),
            self._target_z0,
        ))

        actor_scan, _ = self._scans(self._bed_grid)
        self._fill_kg = self._compute_fill(pos, env)
        self._fill_filt.mul_(1.0 - self._fill_alpha).add_(self._fill_kg, alpha=self._fill_alpha)
        self._spill_filt.mul_(1.0 - self._spill_alpha).add_(self._fill_kg, alpha=self._spill_alpha)
        cut = self._cut_state(actor_scan)

        spill_delta = R.fill_delta_reward(self._spill_filt, self._spill_filt_prev)
        fill_frac = self._fill_kg / c.target_load_kg
        if c.fill_success_mode == "all":
            self._drum_full = R.drums_full(fill_frac, c.fill_success_fraction)
        elif c.fill_success_mode == "mean":
            self._drum_full = fill_frac.mean(dim=-1) >= c.fill_success_fraction
        else:
            self._drum_full = R.front_drum_full(fill_frac, c.fill_success_fraction)

        above = cut["above_volume"]
        self._above_init = torch.where(self._cut_fresh, above, self._above_init)

        # Mass balance, kg:
        #   removal   work-area soil gone from above the plane since the settle
        #   cut_total removal plus soil taken from below the plane
        #   payload   held in the drums, capped at target_load_kg per drum
        #   unheld    |removal - payload|
        removal = ((self._above_init - above) * c.soil_density).clamp_min(0.0)
        cut_total = removal + cut["below_volume"] * c.soil_density
        payload = self._fill_kg.clamp(max=c.target_load_kg).sum(dim=-1)
        unheld = (removal - payload).abs()
        self._removed_kg = removal
        self._cut_total_kg = cut_total
        self._payload_kg = payload
        removal_delta = torch.where(
            self._cut_fresh, torch.zeros_like(removal), removal - self._removal_prev,
        )
        unheld_delta = torch.where(
            self._cut_fresh, torch.zeros_like(unheld), unheld - self._unheld_prev,
        )

        # Shape achieved: residual within tolerance and enough of the cut held.
        residual = above + cut["below_volume"]
        was_done = self._shape_done
        self._shape_done = (
            (residual <= c.shape_success_fraction * c.cut_volume_ref)
            & (payload >= c.capture_min_fraction * cut_total)
            & ~self._cut_fresh
        )
        self._shape_newly_done = self._shape_done & ~was_done
        overcut_delta = R.progress_delta(self._below_prev, cut["below_volume"], self._cut_fresh)

        wheel_cmd_rad = self._forward_cmd / WHEEL_RADIUS

        def finite(t: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        all_vel = finite(d.joint_vel.torch)
        all_tau = finite(d.applied_torque.torch)
        base_lin_vel = finite(p["base_lin_vel"])
        projected_gravity = finite(p["projected_gravity"])
        load_ref = 2.0 * c.target_load_kg

        self._last_terms = {
            "removal": c.w_removal * removal_delta / load_ref,
            "unheld": -c.w_unheld * unheld_delta / load_ref,
            "success": c.w_success * self._shape_newly_done.float(),
            "fail": -c.w_fail * self._failed_now.float(),
            "overcut": c.w_overcut * overcut_delta / c.cut_volume_ref,
            "spill": -c.w_spill * R.spill_penalty(spill_delta) / load_ref,
            "stall": -c.w_stall * R.stall_penalty(wheel_cmd_rad, base_lin_vel[:, 0], WHEEL_RADIUS),
            "drift": -c.w_drift * R.drift_penalty(base_lin_vel, self._forward_cmd),
            "upright": -c.w_upright * R.upright_penalty(projected_gravity),
            "energy": -c.w_energy * R.energy_penalty(all_tau, all_vel),
            "action_rate": -c.w_action_rate * R.action_rate_penalty(self._actions, self._prev_actions),
            "time": -c.w_time * torch.ones(self.num_envs, device=self.device),
        }
        # Envs whose articulation went non-finite this step are paid nothing.
        self._last_terms = {
            k: torch.where(self._nonfinite_now, torch.zeros_like(v), v)
            for k, v in self._last_terms.items()
        }
        reward = L.add_all(self._last_terms)
        if not torch.isfinite(reward).all():
            bad = {
                k: int((~torch.isfinite(v)).sum())
                for k, v in self._last_terms.items()
                if not torch.isfinite(v).all()
            }
            raise RuntimeError(
                f"nonfinite reward from terms {bad}; "
                f"envs={self.num_envs} step={int(self.episode_length_buf.max())}"
            )
        self._spill_filt_prev[:] = self._spill_filt
        self._removal_prev[:] = removal
        self._unheld_prev[:] = unheld
        self._below_prev[:] = cut["below_volume"]
        self._settle_left = (self._settle_left - 1).clamp_min(0)
        self._cut_fresh[:] = self._settle_left > 0
        self._since_settle = torch.where(self._cut_fresh, torch.zeros_like(self._since_settle), self._since_settle + 1)
        return reward

    # ------------------------------------------------------------------
    # terminations
    # ------------------------------------------------------------------

    def _failed(self) -> torch.Tensor:
        """Base failures, plus (with spawn_on_bed) the chassis leaving the bed
        inset by bed_margin_x/y, or a particle beyond particle_escape_m."""
        bad = super()._failed()
        c = self.cfg
        if not c.spawn_on_bed:
            return bad
        xy = self.robot.data.root_pos_w.torch[:, :2] - self.scene.env_origins[:, :2]
        lo = xy.new_tensor([c.bed_x[0] + c.bed_margin_x, c.bed_y[0] + c.bed_margin_y])
        hi = xy.new_tensor([c.bed_x[1] - c.bed_margin_x, c.bed_y[1] - c.bed_margin_y])
        self._off_bed = ((xy < lo) | (xy > hi)).any(dim=-1)
        self._escaped = self._particle_max_m > c.particle_escape_m
        return bad | self._off_bed | self._escaped

    def _nonfinite_state(self) -> torch.Tensor:
        """(E,) True where root pose, root velocity or joint state is non-finite."""
        d = self.robot.data
        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for t in (
            d.root_pos_w.torch,
            d.root_quat_w.torch,
            d.root_lin_vel_w.torch,
            d.root_ang_vel_w.torch,
            d.joint_pos.torch,
            d.joint_vel.torch,
        ):
            bad |= ~torch.isfinite(t.reshape(self.num_envs, -1)).all(dim=-1)
        return bad

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Failures, non-finite envs and a finished pass terminate; shape_done
        does only with terminate_early. A non-finite env is not counted as failed."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        failed = self._failed()
        self._nonfinite_now = self._nonfinite_state()
        heading = torch.stack([torch.cos(self._work_yaw), torch.sin(self._work_yaw)], dim=-1)
        front = self._drum_poses()[0][:, 0, :2]
        self._pass_done = ((front - self._target_xy) * heading).sum(dim=-1) >= self.cfg.work_area_length
        terminated = failed | self._nonfinite_now | self._pass_done
        if self.cfg.terminate_early:
            terminated = terminated | self._shape_done
        self._failed_now = failed & ~self._nonfinite_now
        self._off_bed = self._off_bed & ~self._nonfinite_now

        bad_envs = tuple(self._nonfinite_now.nonzero().flatten().tolist())
        if bad_envs != self._nonfinite_last:
            step = int(self.common_step_counter)
            if bad_envs:
                print(f"[env] articulation state non-finite in env(s) {list(bad_envs)} "
                      f"from step {step}; terminating them to reset", flush=True)
            else:
                print(f"[env] env(s) {list(self._nonfinite_last)} finite again after "
                      f"{step - self._nonfinite_since} steps", flush=True)
            self._nonfinite_last, self._nonfinite_since = bad_envs, step
        return terminated, time_out

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def _reset_particle_history(self, env_ids: torch.Tensor) -> None:
        """Clear the MPM per-particle history (elastic strain, plastic state,
        stress, velocity gradient) in these envs' Newton worlds.

        Env i is Newton world i; the extra last slot is the global world -1.
        Joint state is untouched.
        """
        manager = self.sim.physics_manager
        solver = manager._solver
        worlds = int(solver.model.world_count)
        # A one-world MPM grid has no per-env partition and refuses a masked
        # reset; there the reset is the full one, as in Isaac Lab's coupler.
        world_mask = None
        if worlds > 1:
            selected = np.zeros(worlds + 1, dtype=bool)
            selected[env_ids.cpu().numpy()] = True
            world_mask = wp.array(selected, dtype=wp.bool, device=solver.model.device)
        solver.reset(manager.backend.state_0, world_mask=world_mask, flags=newton.StateFlags.PARTICLE)

    def _reset_idx(self, env_ids: torch.Tensor):
        exits = self._exit_stats(env_ids)
        exits["Episode/shape_done"] = float(self._shape_done[env_ids].float().mean())
        exits["Episode/pass_done"] = float(self._pass_done[env_ids].float().mean())
        exits["Episode/drum_full"] = float(self._drum_full[env_ids].float().mean())
        exits["Episode/removed_kg"] = float(self._removed_kg[env_ids].mean())
        exits["Episode/payload_kg"] = float(self._payload_kg[env_ids].mean())
        exits["Episode/cut_total_kg"] = float(self._cut_total_kg[env_ids].mean())
        # Soil above the plane when the settle ended, kg.
        exits["Episode/above_init_kg"] = float(
            (self._above_init[env_ids] * self.cfg.soil_density).mean()
        )
        exits["Episode/failed"] = float(self._failed_now[env_ids].float().mean())
        exits["Episode/nonfinite"] = float(self._nonfinite_now[env_ids].float().mean())
        exits["Episode/off_bed"] = float(self._off_bed[env_ids].float().mean())
        exits["Episode/escaped"] = float(self._escaped[env_ids].float().mean())
        exits["Episode/particle_max_m"] = float(self._particle_max_m[env_ids].max())
        super()._reset_idx(env_ids)

        n, dev, c = env_ids.numel(), self.device, self.cfg

        def jitter(scale: float) -> torch.Tensor:
            return (torch.rand(n, device=dev) * 2.0 - 1.0) * scale

        yaw = jitter(c.spawn_yaw_jitter)
        xy_off = torch.stack([jitter(c.spawn_x_jitter), jitter(c.spawn_y_jitter)], dim=-1)
        arm = c.arm_start_angle + jitter(c.arm_start_jitter)
        # Rotor phase, uniform over one pocket.
        drum = torch.rand(n, device=dev) * (2.0 * math.pi / ROTOR_VANES)
        spawn_xy = self._reset_robot(env_ids, yaw, arm, xy_off, drum)
        self._arm_start[env_ids] = arm
        self._work_yaw[env_ids] = yaw
        self._since_settle[env_ids] = 0
        self._pass_done[env_ids] = False

        # Soil is re-laid every soil_reset_every episodes, and whenever a
        # particle escaped or the articulation went non-finite.
        self._soil_age[env_ids] += 1
        broken = self._escaped[env_ids] | self._nonfinite_now[env_ids]
        stale = env_ids[(self._soil_age[env_ids] >= c.soil_reset_every) | broken]
        if stale.numel() > 0:
            self.soil.reset(stale)
            self._reset_particle_history(stale)
            self._soil_age[stale] = 0
            self._bed_grid[stale] = c.bed_top

        self._fill_kg[env_ids] = 0.0
        self._fill_filt[env_ids] = 0.0
        self._spill_filt[env_ids] = 0.0
        self._spill_filt_prev[env_ids] = 0.0
        self._fill_bias[env_ids] = (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * c.fill_sensor_bias

        lo, hi = c.cut_depth_range
        depth = lo + (hi - lo) * torch.rand(n, device=dev)
        anchor_xy, surface = self._drum_spawn_ground(env_ids, spawn_xy, yaw, arm)
        floor = self.scene.env_origins[env_ids, 2] + BED_FLOOR_Z
        self._target_xy[env_ids] = anchor_xy
        # Provisional; _get_rewards re-places it during the settle.
        self._target_z0[env_ids] = (surface - depth).clamp_min(floor)
        self._target_depth[env_ids] = depth

        # Gradient sampled in the body frame (x forward) and rotated to world.
        grad = torch.stack([
            jitter(c.cut_gradient_max),
            jitter(c.cut_gradient_max * c.cut_lateral_fraction),
        ], dim=-1)
        ramp = (torch.rand(n, device=dev) < c.cut_ramp_fraction).unsqueeze(-1)
        self._target_grad[env_ids] = R.body_to_world_xy(
            torch.where(ramp, grad, torch.zeros_like(grad)), yaw
        )
        self._set_work_area(env_ids, anchor_xy, yaw)

        self._above_init[env_ids] = 0.0
        self._removal_prev[env_ids] = 0.0
        self._unheld_prev[env_ids] = 0.0
        self._below_prev[env_ids] = 0.0
        self._lead[env_ids] = 0
        self._cut_fresh[env_ids] = True
        self._settle_left[env_ids] = self._settle_steps
        self._shape_done[env_ids] = False
        self._shape_newly_done[env_ids] = False
        self._drum_full[env_ids] = False
        self._failed_now[env_ids] = False
        self._off_bed[env_ids] = False
        self._escaped[env_ids] = False
        self._particle_max_m[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        self.extras["log"] = self._log.flush(env_ids)
        self.extras["log"].update(exits)
