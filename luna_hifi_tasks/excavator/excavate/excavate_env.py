# excavate_env.py
#
# Fill both drums from an MPM regolith bed. Actions are [forward, yaw, boom,
# drum, shroud]; boom, drum and shroud are single commands applied to both
# ends, which is the counter-rotating, reaction-cancelling dig.
#
# The shroud is the fixed outer half of each drum, carrying the inlet. It
# hinges on the drum axis with its own actuator, so the policy aims the inlet
# independently of the arm: down into the cut to load, turned up to hold,
# turned over the hopper to dump.
#
# The reward is captured soil, the mass of particles inside each drum's bore
# in the drum's own frame, plus progress toward the commanded cut plane.
#
# Step order in DirectRLEnv: _pre_physics_step -> _apply_action (x decimation)
# -> _get_dones -> _get_rewards -> _reset_idx -> _get_observations. Fill is
# computed once per step in _get_rewards and cached for the observation.

from __future__ import annotations

import math

import torch

from isaaclab_newton.assets import MPMObject

from ..excavator import ARM_LEN, PIVOT_X, ROTOR_VANES, WHEEL_RADIUS
from ..excavator_cfg import MAX_DRUM_SPEED
from ..excavator_env_base import ExcavatorEnvBase
from ..mdp import rewards as R
from ..mdp.sensors import drum_fill_mass, mpm_grid_particle_mass, mpm_particle_state, soil_heightmap
from ..mdp.observations import DIG_SCAN_CELL
from ..mdp.terrain import (
    degrade_scan,
    dig_scan_pattern,
    nav_far_pattern,
    nav_near_pattern,
    sample_height_grid,
    scan_from_heightfield,
    scan_points_world,
)
from .excavate_env_cfg import (
    BED_FLOOR_Z,
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
        # __post_init__ derives the bed, the grid capacities and the MPM
        # material from the primitive fields, and Hydra applies its overrides
        # after it. 
        n = int(cfg.scene.num_envs)
        assert n <= cfg.max_num_envs, (
            f"num_envs={n} exceeds max_num_envs={cfg.max_num_envs}, the ceiling this "
            "config declares. Raise max_num_envs in the cfg and check the grid budget."
        )
        cfg.max_num_envs = n
        cfg.__post_init__()
        super().__init__(cfg, render_mode, **kwargs)

        E, dev = self.num_envs, self.device
        self._actions = torch.zeros(E, 5, device=dev)
        self._prev_actions = torch.zeros(E, 5, device=dev)

        self._particle_mass = mpm_grid_particle_mass(cfg.scene.soil.spawn)
        self._fill_kg = torch.zeros(E, 2, device=dev)
        self._fill_prev_kg = torch.zeros(E, 2, device=dev)

        self._dig_pattern = dig_scan_pattern(dev)
        # The critic's window matches navigate's, so one critic layout serves both tiers.
        self._far_pattern = nav_far_pattern(dev)
        self._near_pattern = nav_near_pattern(dev)

        # Scan cells over the drum, in the drum's frame. 
        self._footprint = (
            (self._dig_pattern[:, 0].abs() <= FOOTPRINT_HALF_X)
            & (self._dig_pattern[:, 1].abs() <= FOOTPRINT_HALF_Y)
        )
        self._cell_area = DIG_SCAN_CELL ** 2

        # Env-local xy of every bed-grid cell centre, so the commanded plane
        # and the work-area mask can be evaluated over the whole bed.
        gx = cfg.bed_grid_lower[0] + (torch.arange(cfg.bed_grid_nx, device=dev) + 0.5) * cfg.voxel_size
        gy = cfg.bed_grid_lower[1] + (torch.arange(cfg.bed_grid_ny, device=dev) + 0.5) * cfg.voxel_size
        self._grid_xy = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1)
        self._grid_cell_area = cfg.voxel_size ** 2
        # Which of those cells this episode's cut is responsible for.
        self._work_mask = torch.zeros(
            E, cfg.bed_grid_ny, cfg.bed_grid_nx, dtype=torch.bool, device=dev
        )
        self._work_done = torch.zeros(E, device=dev)

        # The commanded cut, as a plane in the world frame:
        #   z(X, Y) = _target_z0 + _target_grad . ([X, Y] - _target_xy)
        self._target_xy = torch.zeros(E, 2, device=dev)
        self._target_z0 = torch.zeros(E, device=dev)
        self._target_grad = torch.zeros(E, 2, device=dev)
        # The episode's exit: the commanded plane is cut. Carrying the spoil
        # is not the deliverable -- a pass of the commanded length yields
        # several times what the pair can hold, so the drums filling is
        # recorded for the log and does not end the pass.
        self._shape_done = torch.zeros(E, dtype=torch.bool, device=dev)
        # True only on the step the shape is first met, so success pays once.
        self._shape_newly_done = torch.zeros(E, dtype=torch.bool, device=dev)
        self._drum_full = torch.zeros(E, dtype=torch.bool, device=dev)
        # Set in _get_dones, which runs before _get_rewards in the same step.
        self._failed_now = torch.zeros(E, dtype=torch.bool, device=dev)
        # Which failures were the machine running out of bed, for the log.
        self._off_bed = torch.zeros(E, dtype=torch.bool, device=dev)
        # Which failures were a particle thrown clear of the env, for the log.
        self._escaped = torch.zeros(E, dtype=torch.bool, device=dev)
        # Furthest any particle got from its env origin this episode. The MPM
        # grid's point mask drops non-finite positions but nothing else, so a
        # particle flung to a large finite position is still packed into the
        # index grid and the NanoVDB tree has to span out to it: one escapee
        # 84 km out exceeds 16,384 lower nodes, and 671 km exceeds 4,096 upper
        # ones, which no capacity can be provisioned against.
        self._particle_max_m = torch.zeros(E, device=dev)
        self._last_terms: dict[str, torch.Tensor] = {}
        # Last step's bed, which a reset samples for the surface the drum
        # comes down on.
        self._bed_grid = torch.full(
            (E, cfg.bed_grid_ny, cfg.bed_grid_nx), cfg.bed_top, device=dev
        )
        # Episodes since this env's soil was last returned to its spawn cells,
        # staggered so the envs do not all refresh on the same episode.
        self._soil_age = torch.randint(0, max(cfg.soil_reset_every, 1), (E,), device=dev)
        # Soil still above target last step, and a mask for an episode's
        # first step, which has no previous value to difference against.
        self._above_prev = torch.zeros(E, device=dev)
        # The work area's soil volume at the start of the episode. Removal is
        # measured against this rather than accumulated from per-step deltas:
        # above_volume rises again when soil slumps back in, and a per-step
        # delta rectified at zero would ratchet that noise into free progress.
        self._above_init = torch.zeros(E, device=dev)
        # Telescoping references for the two mass-balance terms, so an episode
        # total reads directly in kilograms.
        self._removal_prev = torch.zeros(E, device=dev)
        self._unheld_prev = torch.zeros(E, device=dev)
        # The balance itself, in kilograms, for the log. Declared here because
        # a reset can flush the log before the first reward step has run.
        self._removed_kg = torch.zeros(E, device=dev)
        self._cut_total_kg = torch.zeros(E, device=dev)
        self._payload_kg = torch.zeros(E, device=dev)
        # Which drum leads, 0 front and 1 rear, latched against the commanded
        # drive so it does not flip while the machine is near a standstill.
        self._lead = torch.zeros(E, dtype=torch.long, device=dev)

        self._below_prev = torch.zeros(E, device=dev)
        self._cut_fresh = torch.ones(E, dtype=torch.bool, device=dev)
        # Steps of settling still owed. _cut_fresh is this being non-zero, so
        # one step of it reproduces the old behaviour exactly.
        self._settle_steps = max(
            1, round(cfg.settle_seconds / (cfg.sim.dt * cfg.decimation))
        )
        self._settle_left = torch.full(
            (E,), self._settle_steps, dtype=torch.long, device=dev
        )

        # Load sensor: first-order lag plus a per-episode calibration bias.
        self._fill_filt = torch.zeros(E, 2, device=dev)
        self._fill_filt_prev = torch.zeros(E, 2, device=dev)
        self._fill_bias = torch.zeros(E, 2, device=dev)
        step_dt = cfg.sim.dt * cfg.decimation
        self._fill_alpha = step_dt / (cfg.fill_sensor_tau + step_dt)
        # spill's own filter, slow enough to be blind to vane passage.
        self._spill_filt = torch.zeros(E, 2, device=dev)
        self._spill_filt_prev = torch.zeros(E, 2, device=dev)
        self._spill_alpha = step_dt / (cfg.spill_filter_tau + step_dt)

        self._log = R.TermLogger(
            ["removal", "unheld", "success", "fail", "overcut", "spill",
             "stall", "drift", "upright", "energy", "action_rate", "time"],
            E, dev,
        )
        # The __post_init__ above spawned the soil from these fields.
        mismatch = cfg.soil_material_mismatch()
        if mismatch is not None:
            raise RuntimeError(
                f"soil cfg fields do not match the spawned MPM material ({mismatch})"
            )

        print(DIG_OBS.describe())
        print(DIG_CRITIC.describe())
        mat = cfg.scene.soil.spawn.material
        print(
            f"[excavate] particle mass {self._particle_mass:.4f} kg, "
            f"target load {cfg.target_load_kg:.1f} kg "
            f"(swept volume would hold {cfg.drum_capacity_kg:.1f}), "
            f"cut ref {cfg.cut_volume_ref:.4f} m3, "
            f"bed grid {cfg.bed_grid_nx} x {cfg.bed_grid_ny} @ {cfg.voxel_size:.3f} m"
        )
        # Off the material the solver was handed.
        print(
            f"[excavate] soil AS SPAWNED: density {mat.density:.0f} kg/m3, "
            f"friction {mat.friction:.2f}, cohesion (yield_stress) {mat.yield_stress:.0f} Pa, "
            f"yield_pressure {mat.yield_pressure:.3g} Pa"
        )

    # ------------------------------------------------------------------

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
        a = self._actions
        lo, hi = self.cfg.arm_range
        slo, shi = self.cfg.shroud_range
        self._apply_drive(a[:, 0], a[:, 1])
        self._update_lead()
        self._apply_arms(lo + 0.5 * (a[:, 2] + 1.0) * (hi - lo))
        self._apply_drums(a[:, 3] * MAX_DRUM_SPEED)
        self._apply_shrouds(slo + 0.5 * (a[:, 4] + 1.0) * (shi - slo))

    # ------------------------------------------------------------------
    # soil sensing: these are the methods that gather information about the soil state and parse it for the agent's observations
    # ------------------------------------------------------------------

    def _update_lead(self) -> None:
        """Latch which drum is cutting from the sign of the commanded drive.

        _forward_cmd is the drive in m/s and signed, so its sign is the
        policy's own intent rather than a noisy measurement. Inside the
        deadband the previous choice stands: without that the scan would swing
        180 degrees on any step the command grazed zero.
        """
        v = self._forward_cmd
        db = self.cfg.dig_lead_deadband
        front = torch.zeros_like(self._lead)
        rear = torch.ones_like(self._lead)
        self._lead = torch.where(v > db, front, torch.where(v < -db, rear, self._lead))

    def _lead_drum(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(E, 3) position of the leading drum and (E,) the yaw to scan along.

        pi for the rear drum, which is a 180-degree copy: its scan window has
        to point out the back of the machine, at ground it has not cut yet.
        """
        pos = self._drum_poses()[0]
        idx = self._lead.view(-1, 1, 1).expand(-1, 1, pos.shape[-1])
        return pos.gather(1, idx).squeeze(1), self._lead.to(pos.dtype) * math.pi

    def _particles(self) -> tuple[torch.Tensor, torch.Tensor]:
        return mpm_particle_state(self.soil)

    def _compute_fill(self, pos: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        """(E, 2) kg of soil inside the front and rear drum bores."""
        dpos, dquat = self._drum_poses()
        out = torch.zeros(self.num_envs, 2, device=self.device)
        for i in range(2):
            out[:, i] = drum_fill_mass(
                pos, env, self._particle_mass, dpos[:, i], dquat[:, i],
                BORE_RADIUS, BORE_HALF_LEN, self.num_envs,
            )
        return out

    def _bed_heightmap(self, pos: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        return soil_heightmap(
            pos, env, self.scene.env_origins,
            c.bed_grid_lower, c.voxel_size, c.bed_grid_nx, c.bed_grid_ny, BED_FLOOR_Z,
        )

    def _scans(self, bed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(actor scan at the front drum, critic scan around the chassis).
        One rasterisation, sampled three times."""
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
        # The critic reads the chassis, which has no leading end.
        critic = torch.cat(
            [sample(chassis, self._far_pattern), sample(chassis, self._near_pattern)], dim=-1
        )
        return actor, critic

    def _cut_state(self, actor_scan: torch.Tensor) -> dict[str, torch.Tensor]:
        """The cut command and how far the ground is from meeting it.

        Every height here is in the scan's own frame -- relative to the front
        drum, clipped to scan_clip -- so target and terrain are directly
        comparable and the drum's own height cancels out of depth_error.

        The cut command is represented by a target plane defined by a reference height and a gradient. 
        The plane specifies the desired soil surface that the excavator should achieve, and the residual 
        indicates how far the actual ground is from this target.
        """
        clip = self.cfg.scan_clip
        front, lead_yaw = self._lead_drum()
        drum_z = front[:, 2]

        floor = self.scene.env_origins[:, 2] + BED_FLOOR_Z

        def plane_at(xy: torch.Tensor) -> torch.Tensor:
            """World z of the commanded plane at world xy, clamped to the
            hard floor. Trailing dims free.

            At the 0.30 gradient cap an unclamped plane drops below the floor
            after about half a metre of travel.
            """
            nd = xy.dim() - 2
            rel = xy - self._target_xy.view(-1, *([1] * nd), 2)
            z = self._target_z0.view(-1, *([1] * nd)) + (
                rel * self._target_grad.view(-1, *([1] * nd), 2)
            ).sum(dim=-1)
            return torch.maximum(z, floor.view(-1, *([1] * nd)))

        # The plane sampled at the cells the scan was taken at, in the
        # convention every scan backend uses: reference height minus ground
        # height, so a lower surface reads larger.
        pts = scan_points_world(
            front, self.robot.data.root_quat_w.torch, self._dig_pattern, lead_yaw
        )[:, self._footprint]
        t = (drum_z.unsqueeze(-1) - plane_at(pts)).clamp(-clip, clip)
        level = (drum_z - plane_at(front[:, :2])).clamp(-clip, clip)
        # Same frame as the scan: with the rear drum leading, body +x points
        # back along the direction of travel, so the gradient the machine is
        # about to climb flips with it.
        grad_b = R.world_to_body_xy(self._target_grad, self._base_yaw() + lead_yaw)
        foot = actor_scan[:, self._footprint]
        # t - foot is (drum - target) - (drum - ground) = ground - target, so
        # the drum's own height cancels. Positive is soil still standing above
        # the commanded plane.
        residual = t - foot

        # Progress is measured over the work area, a patch fixed in the world
        # for the episode, so the sum of the per-step differences is the total
        # volume the machine removed from it.
        bed_z = self._bed_grid + self.scene.env_origins[:, 2].view(-1, 1, 1)
        plane_z = self._plane_over_grid()
        # A cell with no particles rasterises to BED_FLOOR_Z, the same height
        # as one excavated to bedrock, so cells at the floor are excluded from
        # both volumes.
        floor = (self.scene.env_origins[:, 2] + BED_FLOOR_Z).view(-1, 1, 1)
        soil = self._work_mask & (bed_z > floor + 0.5 * self.cfg.voxel_size)
        cell_residual = torch.where(soil, bed_z - plane_z, torch.zeros_like(bed_z))
        area = self._grid_cell_area
        cells = self._work_mask.flatten(1).sum(-1).clamp_min(1)
        done = (self._work_mask & (bed_z <= plane_z)).flatten(1).sum(-1)
        # Bare floor inside the work area counts as finished.
        self._work_done = (done / cells).float()
        return {
            "target_level": level.unsqueeze(-1),
            "grad_forward": grad_b[:, 0:1],
            "grad_lateral": grad_b[:, 1:2],
            # Local, under the drum: what the policy servos the boom on.
            "depth_error": residual.mean(dim=-1, keepdim=True),
            # Global, over the work area: how much of the job is done.
            "cut_progress": self._work_done.unsqueeze(-1),
            # m3 of soil still above the target, and taken below it
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
        """(world xy, world z) of the soil surface where the front drum comes
        down. The xy is the anchor the commanded plane is measured from.

        Sampled from the bed as it stood at the end of the previous episode,
        so a carried-over bed gets a target below the ground it has left.
        """
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

    def _plane_over_grid(self) -> torch.Tensor:
        """World z of the commanded plane at every bed-grid cell. (E, ny, nx)."""
        rel = self._grid_xy.unsqueeze(0) - self._target_xy.view(-1, 1, 1, 2)
        z = self._target_z0.view(-1, 1, 1) + (
            rel * self._target_grad.view(-1, 1, 1, 2)
        ).sum(dim=-1)
        floor = (self.scene.env_origins[:, 2] + BED_FLOOR_Z).view(-1, 1, 1)
        return torch.maximum(z, floor)

    def _set_work_area(
        self,
        env_ids: torch.Tensor,
        anchor_xy: torch.Tensor,
        yaw: torch.Tensor,
    ) -> None:
        """Mark the bed-grid cells this episode's cut is responsible for: a
        rectangle at the anchor, aligned with the approach heading."""
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
        """Estimated load as a fraction of target_load_kg, (E, 2)."""
        c = self.cfg
        reading = self._fill_filt * (1.0 + self._fill_bias)
        noise = torch.randn_like(reading) * (c.fill_sensor_noise * reading.abs() + c.fill_sensor_abs_kg)
        return ((reading + noise) / c.target_load_kg).clamp(-0.5, 3.0)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    # Reported once per field, because the thing that stops a training run is
    # "observation group 'policy' contains NaN values" and that names the
    # group, not which of nineteen tensors went bad. A per-field isnan over
    # four envs is free next to a 1.2 s step.
    _nan_seen = set()

    def _report_nonfinite(self, group: str, fields: dict) -> None:
        for name, t in fields.items():
            key = f"{group}.{name}"
            if key in self._nan_seen:
                continue
            bad = ~torch.isfinite(t)
            if not bool(bad.any()):
                continue
            self._nan_seen.add(key)
            idx = bad.nonzero()[:4].tolist()
            print(
                f"[obs] NON-FINITE in {key}: {int(bad.sum())} of {t.numel()} "
                f"entries, shape {tuple(t.shape)}, first at {idx}, "
                f"step {int(self.common_step_counter)}",
                flush=True,
            )

    def _get_observations(self) -> dict:
        c = self.cfg
        p = self._proprio()
        pos, env = self._particles()
        actor_scan, critic_scan = self._scans(self._bed_heightmap(pos, env))
        cut = self._cut_state(actor_scan)
        policy_fields = {
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
        }
        self._report_nonfinite("policy", policy_fields)
        policy = DIG_OBS.assemble(policy_fields)

        wheel_vel = self.robot.data.joint_vel.torch
        v_fwd = p["base_lin_vel"][:, 0]
        slip = torch.stack([
            R.wheel_slip(wheel_vel[:, self._left_ids], v_fwd, WHEEL_RADIUS),
            R.wheel_slip(wheel_vel[:, self._right_ids], v_fwd, WHEEL_RADIUS),
        ], dim=-1)
        critic_fields = {
            "terrain_scan_true": critic_scan,
            "drum_fill_mass": self._fill_kg,
            "base_lin_vel_w": self.robot.data.root_lin_vel_w.torch,
            "slip": slip,
        }
        self._report_nonfinite("critic", critic_fields)
        critic = DIG_CRITIC.assemble(critic_fields)
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
        actor_scan, _ = self._scans(self._bed_grid)
        self._fill_kg = self._compute_fill(pos, env)
        self._fill_filt.mul_(1.0 - self._fill_alpha).add_(self._fill_kg, alpha=self._fill_alpha)
        self._spill_filt.mul_(1.0 - self._spill_alpha).add_(self._fill_kg, alpha=self._spill_alpha)
        cut = self._cut_state(actor_scan)

        # spill rectifies its input, so it sums the downward variation of
        # whatever it is given, and takes the lagged reading, which moves on
        # sustained loss and not on the particles crossing the bore boundary
        # each step. The payload below is capped at the target load per drum:
        # uncapped, progress counted every kilogram the drums ended up
        # holding, and at iteration 198 that was 172 kg in a pair that carries
        # about 52 at normal packing, reached by burying the drums and
        # compacting the soil to half of solid, which needed a cut below the
        # plane. The cap pays for filling, not for packing.
        cap = c.target_load_kg
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

        # --- the mass balance ----------------------------------------------
        # removal: mass gone from the commanded work area since the episode
        #   began, measured against the opening surface rather than summed
        #   from per-step deltas.
        # payload: mass the drums hold, per drum capped at the target load so
        #   that burying and compacting them buys nothing -- an earlier
        #   uncapped version paid for 172 kg in a pair that carries about 52,
        #   which it reached by digging below the plane.
        # unheld: the gap between removal and payload, both ways. Displacing
        #   the layer leaves payload at zero and cancels the removal payment;
        #   holding more than the work area gave up above the plane is charged
        #   for the excess, which is what prices soil taken from under the
        #   plane -- at iteration 38 the drums held 20.2 kg against a removal
        #   of 16.2 with overcut at -17.9, the machine digging down rather
        #   than along.
        #   Deliberately not measured against cut_total below: counting the
        #   hole on this side of the balance lets the payload match it and
        #   cancels the charge, which scores digging down at +22.9 where
        #   removal scores it at -1.4. scripts/balance_check.py holds both.
        # cut_total: every kilogram the work area gave up, hole included. Only
        #   the success gate reads it, where adding the hole can only make the
        #   requirement harder to meet.
        removal = ((self._above_init - above) * c.soil_density).clamp_min(0.0)
        cut_total = removal + cut["below_volume"] * c.soil_density
        payload = self._fill_kg.clamp(max=cap).sum(dim=-1)
        unheld = (removal - payload).abs()
        self._removed_kg = removal
        self._cut_total_kg = cut_total
        self._payload_kg = payload
        removal_delta = torch.where(
            self._cut_fresh, torch.zeros_like(removal),
            removal - self._removal_prev,
        )
        unheld_delta = torch.where(
            self._cut_fresh, torch.zeros_like(unheld),
            unheld - self._unheld_prev,
        )

        # Shape achieved: the ground matches the commanded plane both ways,
        # the spoil is accounted for, after at least one step. Without the
        # capture test the residual alone was met by shoving the layer out of
        # the work area, which at iteration 18 of the previous run cleared
        # 24.8 kg and kept 0.34 of it.
        residual = above + cut["below_volume"]
        was_done = self._shape_done
        self._shape_done = (
            (residual <= c.shape_success_fraction * c.cut_volume_ref)
            & (payload >= c.capture_min_fraction * cut_total)
            & ~self._cut_fresh
        )
        self._shape_newly_done = self._shape_done & ~was_done
        # overcut charges on the growth of the hole below the plane: a signed
        # delta, telescoping to the volume the episode ended with.
        overcut_delta = R.progress_delta(
            self._below_prev, cut["below_volume"], self._cut_fresh
        )

        wheel_cmd_rad = self._forward_cmd / WHEEL_RADIUS
        # _failed has already run this step and nonfinite_state has flagged a
        # diverged env, so it is resetting regardless and its fail charge is
        # the signal that matters. Zeroing the state the per-step terms read
        # keeps one blown-up env from ending the run: a NaN anywhere in the
        # reward vector stops the trainer outright. The raise below still
        # fires for a nonfinite term that is not explained by diverged state,
        # which would be a reward bug rather than a sim one.
        def finite(t: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        all_vel = finite(d.joint_vel.torch)
        all_tau = finite(d.applied_torque.torch)
        base_lin_vel = finite(p["base_lin_vel"])
        projected_gravity = finite(p["projected_gravity"])
        load_ref = 2.0 * c.target_load_kg

        # Signed weighted terms, kept so a scripted run can read a step
        # reward term by term.
        self._last_terms = {
            "removal": c.w_removal * removal_delta / load_ref,
            "unheld": -c.w_unheld * unheld_delta / load_ref,
            # Once per episode, on the step the shape is first met. _get_dones
            # runs before _get_rewards, so the flag set here is only seen by the
            # termination on the following step -- and the reward for that step
            # paid it a second time. Navigate's goal bonus had the same shape.
            # A transition latch rather than the termination itself, so it is
            # still paid once when terminate_early is off and nothing ends.
            "success": c.w_success * self._shape_newly_done.float(),
            # Every other penalty is per-step, so ending an episode early
            # avoids the rest of them. This is the one-time charge that makes
            # tipping over cost more than the stream it escapes.
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
        reward = L.add_all(self._last_terms)
        # The trainer's own check reports only that some reward was NaN, which
        # does not say which of twelve terms produced it. The soil-derived
        # terms are each masked against nonfinite heights already; the state
        # ones are not all covered by _failed's nonfinite_state, which reads
        # pos, root_lin_vel_b and joint_vel but neither applied_torque nor
        # projected_gravity. One reduction per step buys the term name.
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
        self._fill_prev_kg[:] = self._fill_kg
        self._fill_filt_prev[:] = self._fill_filt
        self._spill_filt_prev[:] = self._spill_filt
        self._above_prev[:] = above
        self._removal_prev[:] = removal
        self._unheld_prev[:] = unheld
        self._below_prev[:] = cut["below_volume"]
        self._settle_left = (self._settle_left - 1).clamp_min(0)
        self._cut_fresh[:] = self._settle_left > 0
        return reward

    # ------------------------------------------------------------------
    # terminations
    # ------------------------------------------------------------------

    def _failed(self) -> torch.Tensor:
        """The base failures, plus running out of bed.

        The bed is retained on all four faces, so there is soil right up to
        the plate and nothing to warn the machine it has reached the end. This
        is a geometric test on the chassis origin against the bed faces inset
        by the machine's own half-extents, which needs no contact query and
        cannot be missed by one. Reaching a plate is the end of the episode,
        not something to drive along.
        """
        bad = super()._failed()
        c = self.cfg
        # Only a bed the machine stands on bounds where it may be. Where the
        # bed is a pile reached from outside, the machine is meant to be off it.
        if not c.spawn_on_bed:
            return bad
        xy = self.robot.data.root_pos_w.torch[:, :2] - self.scene.env_origins[:, :2]
        lo = xy.new_tensor([c.bed_x[0] + c.bed_margin_x, c.bed_y[0] + c.bed_margin_y])
        hi = xy.new_tensor([c.bed_x[1] - c.bed_margin_x, c.bed_y[1] - c.bed_margin_y])
        self._off_bed = ((xy < lo) | (xy > hi)).any(dim=-1)
        # A particle thrown clear of its own env. The MPM grid's point mask
        # drops only non-finite positions, so a finite escapee is still packed
        # into the index grid and the NanoVDB tree has to span out to it --
        # the documented cause of the rebuild failures, against which no
        # capacity can be provisioned. It ran at 3.5 m for the whole of two
        # earlier runs and reached 107 m at iteration 38 of the one that dug
        # below the plane, so it tracks the drum being buried and spun rather
        # than being sim noise. The bed reaches 3.4 m from the origin and envs
        # sit 9 m apart, so 15 m is past anything legitimate and far below the
        # scale that breaks the tree. Ending the episode both protects the
        # grid and prices the configuration that throws the soil.
        self._escaped = self._particle_max_m > c.particle_escape_m
        return bad | self._off_bed | self._escaped

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._failed_now = self._failed()
        terminated = self._failed_now | self._shape_done
        if not self.cfg.terminate_early:
            # Every world has to reset on the same step for the shared MPM
            # grid, so nothing ends an episode but the timeout. The flags
            # above are still set, and success and fail still pay.
            terminated = torch.zeros_like(terminated)
        return terminated, time_out

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Read before the flags are cleared below.
        exits = self._exit_stats(env_ids)
        exits["Episode/shape_done"] = float(self._shape_done[env_ids].float().mean())
        exits["Episode/drum_full"] = float(self._drum_full[env_ids].float().mean())
        # In kilograms, so the balance is read rather than divided back out of
        # a reward by a weight and a volume reference.
        exits["Episode/removed_kg"] = float(self._removed_kg[env_ids].mean())
        exits["Episode/payload_kg"] = float(self._payload_kg[env_ids].mean())
        exits["Episode/cut_total_kg"] = float(self._cut_total_kg[env_ids].mean())
        # The reference surface itself, in kilograms of soil standing over the
        # plane when the settle ended. A nominal cut is cut_volume_ref; this
        # drifting far above it means the bed is still consolidating when the
        # episode starts and settle_seconds is too short.
        exits["Episode/above_init_kg"] = float(
            (self._above_init[env_ids] * self.cfg.soil_density).mean()
        )
        exits["Episode/failed"] = float(self._failed_now[env_ids].float().mean())
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
        # Rotor phase, sampled over one pocket so drum_phase is decorrelated
        # across the batch.
        pocket = 2.0 * math.pi / ROTOR_VANES
        drum = torch.rand(n, device=dev) * pocket
        spawn_xy = self._reset_robot(env_ids, yaw, arm, xy_off, drum)

        # Soil returns to its spawn cells every soil_reset_every episodes. In
        # between the env keeps the ground it has worked, which is what makes
        # one env's bed differ from another's.
        self._soil_age[env_ids] += 1
        stale = env_ids[self._soil_age[env_ids] >= c.soil_reset_every]
        if stale.numel() > 0:
            self.soil.reset(stale)
            self._soil_age[stale] = 0
            self._bed_grid[stale] = c.bed_top

        # The drums start clear of the bed, so fill is zero whether or not
        # the soil was reset.
        self._fill_kg[env_ids] = 0.0
        self._fill_prev_kg[env_ids] = 0.0
        self._fill_filt[env_ids] = 0.0
        self._fill_filt_prev[env_ids] = 0.0
        self._spill_filt[env_ids] = 0.0
        self._spill_filt_prev[env_ids] = 0.0
        self._fill_bias[env_ids] = (
            torch.rand(n, 2, device=dev) * 2.0 - 1.0
        ) * c.fill_sensor_bias

        lo, hi = c.cut_depth_range
        depth = lo + (hi - lo) * torch.rand(n, device=dev)
        anchor_xy, surface = self._drum_spawn_ground(env_ids, spawn_xy, yaw, arm)
        floor = self.scene.env_origins[env_ids, 2] + BED_FLOOR_Z
        self._target_xy[env_ids] = anchor_xy
        self._target_z0[env_ids] = (surface - depth).clamp_min(floor)

        # Gradient sampled in the body frame and rotated out, so forward is
        # the direction the machine starts pointing.
        grad = torch.stack([
            jitter(c.cut_gradient_max),
            jitter(c.cut_gradient_max * c.cut_lateral_fraction),
        ], dim=-1)
        ramp = (torch.rand(n, device=dev) < c.cut_ramp_fraction).unsqueeze(-1)
        self._target_grad[env_ids] = R.body_to_world_xy(
            torch.where(ramp, grad, torch.zeros_like(grad)), yaw
        )
        self._set_work_area(env_ids, anchor_xy, yaw)

        self._above_prev[env_ids] = 0.0
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
