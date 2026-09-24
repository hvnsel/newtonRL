# excavate_env.py
#
# Fill both drums from an MPM regolith bed. Actions are [forward, yaw, boom,
# drum, shroud]; boom, drum and shroud are single commands applied to both
# ends, which is the counter-rotating, reaction-cancelling dig.
#
# The shroud is the fixed outer half of each drum, carrying the inlet. It
# hinges on the drum axis with its own actuator, so the policy aims the inlet
# independently of where the arm is pointing: down into the cut to load,
# turned up to hold, turned over the hopper to dump.
#
# The reward is captured soil: mass of particles inside each drum's bore, in
# the drum's own frame. A target-heightmap match is the planner's objective
# one level up.
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
        super().__init__(cfg, render_mode, **kwargs)
        assert self.num_envs <= cfg.max_num_envs, (
            f"num_envs={self.num_envs} exceeds max_num_envs={cfg.max_num_envs}; the MPM "
            "sparse-grid caps were sized for the latter. Raise max_num_envs in the cfg."
        )

        E, dev = self.num_envs, self.device
        self._actions = torch.zeros(E, 5, device=dev)
        self._prev_actions = torch.zeros(E, 5, device=dev)

        self._particle_mass = mpm_grid_particle_mass(cfg.scene.soil.spawn)
        self._fill_kg = torch.zeros(E, 2, device=dev)
        self._fill_prev_kg = torch.zeros(E, 2, device=dev)
        self._success = torch.zeros(E, dtype=torch.bool, device=dev)

        self._dig_pattern = dig_scan_pattern(dev)
        # The critic's window matches navigate's, so one critic layout serves
        # both tiers.
        self._far_pattern = nav_far_pattern(dev)
        self._near_pattern = nav_near_pattern(dev)

        # Scan cells over the drum, in the drum's frame. 4 x 8 of the 16 x 8
        # grid, 0.53 x 0.88 m, against a rotor that sweeps 0.37 x 0.95 m.
        self._footprint = (
            (self._dig_pattern[:, 0].abs() <= FOOTPRINT_HALF_X)
            & (self._dig_pattern[:, 1].abs() <= FOOTPRINT_HALF_Y)
        )
        self._cell_area = DIG_SCAN_CELL ** 2

        # Commanded ground height for this episode, world frame.
        self._target_z = torch.zeros(E, device=dev)
        # Last step's bed, so a reset can sample the surface the drum will
        # actually meet rather than assuming the bed is still undisturbed.
        self._bed_grid = torch.full(
            (E, cfg.bed_grid_ny, cfg.bed_grid_nx), cfg.bed_top, device=dev
        )
        # Episodes since this env's soil was last returned to its spawn cells,
        # staggered so the envs do not all refresh on the same episode.
        self._soil_age = torch.randint(0, max(cfg.soil_reset_every, 1), (E,), device=dev)
        # Soil still above target last step, and a mask for an episode's first
        # step, where there is no previous value to difference against.
        self._above_prev = torch.zeros(E, device=dev)
        self._cut_fresh = torch.ones(E, dtype=torch.bool, device=dev)

        # Load sensor: first-order lag plus a per-episode calibration bias.
        self._fill_filt = torch.zeros(E, 2, device=dev)
        self._fill_bias = torch.zeros(E, 2, device=dev)
        step_dt = cfg.sim.dt * cfg.decimation
        self._fill_alpha = step_dt / (cfg.fill_sensor_tau + step_dt)

        self._log = R.TermLogger(
            ["fill", "depth", "success", "overcut", "spill", "stall", "drift",
             "upright", "energy", "action_rate", "time"],
            E, dev,
        )
        # InteractiveScene has already spawned the soil from
        # cfg.scene.soil.spawn.material. A soil_* field changed after
        # __post_init__ -- which is what a Hydra override does -- never reached
        # that material, so the run would use the old soil while reporting the
        # new number.
        mismatch = cfg.soil_material_mismatch()
        if mismatch is not None:
            raise RuntimeError(
                f"soil cfg fields do not match the spawned MPM material ({mismatch}).\n"
                "The material is built in ExcavatorExcavateEnvCfg.__post_init__ and Hydra "
                "applies overrides after it, so anything setting a soil_* field late must "
                "call cfg.apply_soil_material() before gym.make(). scripts/dig_demo.py and "
                "scripts/run_task.py do this; a custom launcher has to as well."
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
        # Read back off the material the solver got, not off the cfg fields --
        # the whole point is that those two can disagree.
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
        self._apply_arms(lo + 0.5 * (a[:, 2] + 1.0) * (hi - lo))
        self._apply_drums(a[:, 3] * MAX_DRUM_SPEED)
        self._apply_shrouds(slo + 0.5 * (a[:, 4] + 1.0) * (shi - slo))

    # ------------------------------------------------------------------
    # soil sensing
    # ------------------------------------------------------------------

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
        front = self._drum_poses()[0][:, 0]

        def sample(origin, pattern):
            return scan_from_heightfield(
                bed, origin, quat, pattern, self.scene.env_origins,
                c.bed_grid_lower, c.voxel_size, origin[:, 2], clip=c.scan_clip,
            )

        actor = sample(front, self._dig_pattern)
        critic = torch.cat(
            [sample(chassis, self._far_pattern), sample(chassis, self._near_pattern)], dim=-1
        )
        return actor, critic

    def _cut_state(self, actor_scan: torch.Tensor) -> dict[str, torch.Tensor]:
        """The cut command and how far the ground is from meeting it.

        Every height here is in the scan's own frame -- relative to the front
        drum, clipped to scan_clip -- so target and terrain are directly
        comparable and the drum's own height cancels out of depth_error.
        """
        clip = self.cfg.scan_clip
        drum_z = self._drum_poses()[0][:, 0, 2]
        target = (self._target_z - drum_z).clamp(-clip, clip)
        foot = actor_scan[:, self._footprint]
        t = target.unsqueeze(-1)
        return {
            "target_height": target.unsqueeze(-1),
            "depth_error": (foot.mean(dim=-1) - target).unsqueeze(-1),
            "cut_progress": (foot <= t).float().mean(dim=-1, keepdim=True),
            # m3 of soil still above the target, and taken below it
            "above_volume": (foot - t).clamp_min(0.0).sum(dim=-1) * self._cell_area,
            "below_volume": (t - foot).clamp_min(0.0).sum(dim=-1) * self._cell_area,
        }

    def _surface_z(
        self,
        env_ids: torch.Tensor,
        spawn_xy: torch.Tensor,
        yaw: torch.Tensor,
        arm_angle: torch.Tensor,
    ) -> torch.Tensor:
        """World z of the soil surface where the front drum will come down.

        Sampled from the bed as it stood at the end of the previous episode,
        so a carried-over bed gets a target below what it has already lost
        rather than one it has met before the machine moves.
        """
        c = self.cfg
        reach = PIVOT_X + ARM_LEN * torch.cos(arm_angle)
        q = torch.stack([
            spawn_xy[:, 0] + reach * torch.cos(yaw),
            spawn_xy[:, 1] + reach * torch.sin(yaw),
        ], dim=-1).unsqueeze(1)
        h = sample_height_grid(
            self._bed_grid[env_ids], q, self.scene.env_origins[env_ids],
            c.bed_grid_lower, c.voxel_size, c.bed_top,
        )[:, 0]
        return self.scene.env_origins[env_ids, 2] + h

    def _fill_observed(self) -> torch.Tensor:
        """Estimated load as a fraction of target_load_kg, (E, 2)."""
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
            "target_height": cut["target_height"],
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
        self._bed_grid = self._bed_heightmap(pos, env)
        actor_scan, _ = self._scans(self._bed_grid)
        self._fill_kg = self._compute_fill(pos, env)
        self._fill_filt.mul_(1.0 - self._fill_alpha).add_(self._fill_kg, alpha=self._fill_alpha)
        cut = self._cut_state(actor_scan)

        fill_delta = R.fill_delta_reward(self._fill_kg, self._fill_prev_kg)
        fill_frac = self._fill_kg / c.target_load_kg
        if c.fill_success_mode == "all":
            self._success = R.drums_full(fill_frac, c.fill_success_fraction)
        elif c.fill_success_mode == "mean":
            self._success = fill_frac.mean(dim=-1) >= c.fill_success_fraction
        else:
            self._success = R.front_drum_full(fill_frac, c.fill_success_fraction)

        above = cut["above_volume"]
        depth_delta = R.progress_delta(self._above_prev, above, self._cut_fresh)

        wheel_cmd_rad = self._forward_cmd / WHEEL_RADIUS
        all_vel = d.joint_vel.torch
        all_tau = d.applied_torque.torch
        load_ref = 2.0 * c.target_load_kg

        reward = (
            c.w_fill * L.add("fill", fill_delta / load_ref)
            + c.w_depth * L.add("depth", depth_delta / c.cut_volume_ref)
            + c.w_success * L.add("success", self._success.float())
            - c.w_overcut * L.add("overcut", cut["below_volume"] / c.cut_volume_ref)
            - c.w_spill * L.add("spill", R.spill_penalty(fill_delta) / load_ref)
            - c.w_stall * L.add("stall", R.stall_penalty(wheel_cmd_rad, p["base_lin_vel"][:, 0], WHEEL_RADIUS))
            - c.w_drift * L.add("drift", R.drift_penalty(p["base_lin_vel"], self._forward_cmd))
            - c.w_upright * L.add("upright", R.upright_penalty(p["projected_gravity"]))
            - c.w_energy * L.add("energy", R.energy_penalty(all_tau, all_vel))
            - c.w_action_rate * L.add("action_rate", R.action_rate_penalty(self._actions, self._prev_actions))
            - c.w_time * L.add("time", torch.ones(self.num_envs, device=self.device))
        )
        self._fill_prev_kg[:] = self._fill_kg
        self._above_prev[:] = above
        self._cut_fresh[:] = False
        return reward

    # ------------------------------------------------------------------
    # terminations
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._failed() | self._success, time_out

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        n, dev, c = env_ids.numel(), self.device, self.cfg

        def jitter(scale: float) -> torch.Tensor:
            return (torch.rand(n, device=dev) * 2.0 - 1.0) * scale

        yaw = jitter(c.spawn_yaw_jitter)
        xy_off = torch.stack([jitter(c.spawn_x_jitter), jitter(c.spawn_y_jitter)], dim=-1)
        arm = c.arm_start_angle + jitter(c.arm_start_jitter)
        # Rotor phase over one pocket. Every env started at phase 0, which is
        # a variable correlated across the whole batch now that the policy
        # observes it.
        pocket = 2.0 * math.pi / ROTOR_VANES
        drum = torch.rand(n, device=dev) * pocket
        spawn_xy = self._reset_robot(env_ids, yaw, arm, xy_off, drum)

        # Soil returns to its spawn cells every soil_reset_every episodes. In
        # between the env keeps the ground it has worked, which is the only
        # thing that makes one env's bed differ from another's.
        self._soil_age[env_ids] += 1
        stale = env_ids[self._soil_age[env_ids] >= c.soil_reset_every]
        if stale.numel() > 0:
            self.soil.reset(stale)
            self._soil_age[stale] = 0
            self._bed_grid[stale] = c.bed_top

        # The drums start clear of the bed, so fill is genuinely zero here
        # whether or not the soil was reset.
        self._fill_kg[env_ids] = 0.0
        self._fill_prev_kg[env_ids] = 0.0
        self._fill_filt[env_ids] = 0.0
        self._fill_bias[env_ids] = (
            torch.rand(n, 2, device=dev) * 2.0 - 1.0
        ) * c.fill_sensor_bias

        lo, hi = c.cut_depth_range
        depth = lo + (hi - lo) * torch.rand(n, device=dev)
        floor = self.scene.env_origins[env_ids, 2] + BED_FLOOR_Z
        self._target_z[env_ids] = (
            self._surface_z(env_ids, spawn_xy, yaw, arm) - depth
        ).clamp_min(floor)
        self._above_prev[env_ids] = 0.0
        self._cut_fresh[env_ids] = True
        self._success[env_ids] = False
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        self.extras["log"] = self._log.flush(env_ids, self.cfg.episode_length_s)
