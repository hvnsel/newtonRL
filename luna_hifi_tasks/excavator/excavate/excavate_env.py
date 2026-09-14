# excavate_env.py
#
# Fill both drums from an MPM regolith bed. Actions are [forward, yaw, boom,
# drum]; boom and drum are single commands applied to both ends, which is the
# counter-rotating, reaction-cancelling dig the machine was built for and, at
# lunar gravity, the only way it can dig at all (see excavator_cfg.py).
#
# The reward is captured soil, measured directly: mass of particles inside
# each drum's bore, in the drum's own frame. Not a target-heightmap match --
# that is the planner's objective one level up, and at the skill level it is
# both too sparse to learn from and welds the skill to a site plan.
#
# Step order in DirectRLEnv: _pre_physics_step -> _apply_action (x decimation)
# -> _get_dones -> _get_rewards -> _reset_idx -> _get_observations. Fill is
# computed once per step in _get_rewards and cached for the observation.

from __future__ import annotations

import torch

from isaaclab_newton.assets import MPMObject

from ..excavator import WHEEL_RADIUS
from ..excavator_cfg import MAX_DRUM_SPEED
from ..excavator_env_base import ExcavatorEnvBase
from ..mdp import rewards as R
from ..mdp.sensors import drum_fill_mass, mpm_grid_particle_mass, mpm_particle_state, soil_heightmap
from ..mdp.terrain import dig_scan_pattern, nav_scan_pattern, scan_from_heightfield
from .excavate_env_cfg import (
    BED_FLOOR_Z,
    BORE_HALF_LEN,
    BORE_RADIUS,
    DIG_CRITIC,
    DIG_OBS,
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
        self._actions = torch.zeros(E, 4, device=dev)
        self._prev_actions = torch.zeros(E, 4, device=dev)

        self._particle_mass = mpm_grid_particle_mass(cfg.scene.soil.spawn)
        self._fill_kg = torch.zeros(E, 2, device=dev)
        self._fill_prev_kg = torch.zeros(E, 2, device=dev)
        self._success = torch.zeros(E, dtype=torch.bool, device=dev)

        self._dig_pattern = dig_scan_pattern(dev)
        self._nav_pattern = nav_scan_pattern(dev)

        self._log = R.TermLogger(
            ["fill", "success", "stall", "drift", "upright", "idle_drum", "energy", "action_rate", "time"],
            E, dev,
        )
        # By the time this runs, InteractiveScene has already spawned the soil
        # from cfg.scene.soil.spawn.material. If a soil_* field was changed
        # after __post_init__ -- which is exactly what a Hydra override does,
        # and what `env.soil_cohesion=1500` on the command line looks like --
        # that change never reached the material, and the run would use the old
        # soil while reporting the new number. Silent, and indistinguishable
        # from cohesion simply not mattering.
        mismatch = cfg.soil_material_mismatch()
        if mismatch is not None:
            raise RuntimeError(
                f"soil cfg fields do not match the spawned MPM material ({mismatch}).\n"
                "The material is built in ExcavatorExcavateEnvCfg.__post_init__ and Hydra "
                "applies overrides after it, so anything setting a soil_* field late must "
                "call cfg.apply_soil_material() before gym.make(). scripts/dig_demo.py and "
                "scripts/smoke_test.py do this; a custom launcher has to as well."
            )

        print(DIG_OBS.describe())
        print(DIG_CRITIC.describe())
        mat = cfg.scene.soil.spawn.material
        print(
            f"[excavate] particle mass {self._particle_mass:.4f} kg, "
            f"drum capacity {cfg.drum_capacity_kg:.1f} kg, "
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
        self._apply_drive(a[:, 0], a[:, 1])
        self._apply_arms(lo + 0.5 * (a[:, 2] + 1.0) * (hi - lo))
        self._apply_drums(a[:, 3] * MAX_DRUM_SPEED)

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
        One rasterisation, sampled twice."""
        c = self.cfg
        d = self.robot.data
        quat = d.root_quat_w.torch
        dpos, _ = self._drum_poses()
        front = dpos[:, 0]
        actor = scan_from_heightfield(
            bed, front, quat, self._dig_pattern, self.scene.env_origins,
            c.bed_grid_lower, c.voxel_size, front[:, 2], clip=c.scan_clip,
        )
        chassis = d.root_pos_w.torch
        critic = scan_from_heightfield(
            bed, chassis, quat, self._nav_pattern, self.scene.env_origins,
            c.bed_grid_lower, c.voxel_size, chassis[:, 2], clip=c.scan_clip,
        )
        return actor, critic

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        p = self._proprio()
        pos, env = self._particles()
        actor_scan, critic_scan = self._scans(self._bed_heightmap(pos, env))
        fill_frac = self._fill_kg / self.cfg.drum_capacity_kg

        policy = DIG_OBS.assemble({
            "base_lin_vel": p["base_lin_vel"],
            "base_ang_vel": p["base_ang_vel"],
            "projected_gravity": p["projected_gravity"],
            "wheel_vel": p["wheel_vel"],
            "arm_pos": p["arm_pos"],
            "arm_vel": p["arm_vel"],
            "drum_vel": p["drum_vel"],
            "drum_fill": fill_frac,
            "terrain_scan": actor_scan,
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
        self._fill_kg = self._compute_fill(pos, env)
        fill_delta = R.fill_delta_reward(self._fill_kg, self._fill_prev_kg)
        fill_frac = self._fill_kg / c.drum_capacity_kg
        if c.fill_success_mode == "all":
            self._success = R.drums_full(fill_frac, c.fill_success_fraction)
        else:
            self._success = fill_frac.mean(dim=-1) >= c.fill_success_fraction

        wheel_cmd_rad = self._forward_cmd / WHEEL_RADIUS
        drum_vel = d.joint_vel.torch[:, self._drum_ids]
        all_vel = d.joint_vel.torch
        all_tau = d.applied_torque.torch

        reward = (
            c.w_fill * L.add("fill", fill_delta / (2.0 * c.drum_capacity_kg))
            + c.w_success * L.add("success", self._success.float())
            - c.w_stall * L.add("stall", R.stall_penalty(wheel_cmd_rad, p["base_lin_vel"][:, 0], WHEEL_RADIUS))
            - c.w_drift * L.add("drift", R.drift_penalty(p["base_lin_vel"], self._forward_cmd))
            - c.w_upright * L.add("upright", R.upright_penalty(p["projected_gravity"]))
            - c.w_idle_drum * L.add("idle_drum", R.idle_drum_penalty(drum_vel, fill_delta))
            - c.w_energy * L.add("energy", R.energy_penalty(all_tau, all_vel))
            - c.w_action_rate * L.add("action_rate", R.action_rate_penalty(self._actions, self._prev_actions))
            - c.w_time * L.add("time", torch.ones(self.num_envs, device=self.device))
        )
        self._fill_prev_kg[:] = self._fill_kg
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

        n = env_ids.numel()
        yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.spawn_yaw_jitter
        xy_off = torch.zeros(n, 2, device=self.device)
        xy_off[:, 0] = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.spawn_x_jitter
        self._reset_robot(env_ids, yaw, self.cfg.arm_start_angle, xy_off)

        # Every particle back to its spawn position, zero velocity. The drums
        # start clear of the bed, so fill is genuinely zero here.
        self.soil.reset(env_ids)
        self._fill_kg[env_ids] = 0.0
        self._fill_prev_kg[env_ids] = 0.0
        self._success[env_ids] = False
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0

        self.extras["log"] = self._log.flush(env_ids, self.cfg.episode_length_s)
