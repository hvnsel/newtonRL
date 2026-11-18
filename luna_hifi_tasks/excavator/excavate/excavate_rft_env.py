# excavate_rft_env.py
#
# The excavate task with the analytical soil of mdp/rft.py in place of MPM.
# The bed is the height grid _bed_grid and the drums' load is _fill_kg. Each
# time the actions are applied: the soil's loads on both drums go onto the
# rotor and shroud bodies as world-frame wrenches, then the soil inside the
# vane-tip circles moves from the grid into the drums. Observations, rewards
# and resets are the excavate env's.

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils

from ..excavator_cfg import EXCAVATOR_PRIM_REGEX, LUNAR_G
from ..excavator_env_base import ExcavatorEnvBase
from ..mdp import rft
from .excavate_env import ExcavatorExcavateEnv
from .excavate_env_cfg import BED_FLOOR_Z

# The arms and everything on them. Their colliders are off: the drums and
# shrouds meet the soil through rft.py, and only the wheels touch the ground.
ARM_COLLIDERS = EXCAVATOR_PRIM_REGEX + r"/Geometry/chassis/arm_(front|rear)_body/.*"


class ExcavatorExcavateRftEnv(ExcavatorExcavateEnv):

    def _setup_scene(self):
        ExcavatorEnvBase._setup_scene(self)
        off = [sim_utils.UsdPhysicsCollisionCfg(collision_enabled=False)]
        if not sim_utils.apply_collision_properties(ARM_COLLIDERS, off):
            raise RuntimeError(f"could not switch off the colliders matching {ARM_COLLIDERS}")

    def _init_soil(self) -> None:
        c, dev = self.cfg, self.device
        self._rft = rft.Soil(
            density=c.soil_density, friction=c.soil_friction, gravity=LUNAR_G,
            cell=c.voxel_size, lower=c.bed_grid_lower,
        )
        self._patches = rft.drum_patches(device=dev)
        self._cells = rft.cell_centres(self._rft, c.bed_grid_nx, c.bed_grid_ny, device=dev)
        x, y = self._cells[..., 0], self._cells[..., 1]
        on_bed = (x >= c.bed_x[0]) & (x <= c.bed_x[1]) & (y >= c.bed_y[0]) & (y <= c.bed_y[1])
        # A fresh bed: level at bed_top, the floor around it.
        self._fresh_bed = torch.where(on_bed, c.bed_top, BED_FLOOR_Z)
        self._bed_grid[:] = self._fresh_bed
        self._capacity = torch.full((self.num_envs, 2), c.drum_capacity_kg, device=dev)
        # Front rotor, rear rotor, front shroud, rear shroud.
        self._tool_ids = self._drum_body_ids + self._shroud_body_ids
        print(
            f"[excavate] soil: rft.py, density {c.soil_density:.0f} kg/m3, friction {c.soil_friction:.2f} "
            f"-> stress scale {self._rft.zeta:.0f} N/m3; height grid {c.bed_grid_nx} x {c.bed_grid_ny} "
            f"@ {c.voxel_size:.3f} m; drums hold {c.drum_capacity_kg:.1f} kg, target load {c.target_load_kg:.1f} kg, "
            f"cut ref {c.cut_volume_ref:.4f} m3"
        )

    def _apply_action(self):
        super()._apply_action()
        d = self.robot.data
        ids = self._tool_ids
        pos = d.body_link_pos_w.torch[:, ids]                   # (E, 4, 3), on the drum axes
        quat = d.body_link_quat_w.torch[:, ids]
        lin = d.body_link_lin_vel_w.torch[:, ids]
        ang = d.body_link_ang_vel_w.torch[:, ids]
        centre = pos[:, 2:] - self.scene.env_origins[:, None]  # env-local axis points

        shroud_f, shroud_t, rotor_f, rotor_t = rft.drum_loads(
            self._patches, centre, lin[:, 2:], quat[:, 2:], ang[:, 2:], quat[:, :2], ang[:, :2],
            self._bed_grid, self._fill_kg, self._capacity, self._rft,
        )
        self.robot.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=torch.cat([rotor_f, shroud_f], dim=1),
            torques=torch.cat([rotor_t, shroud_t], dim=1),
            positions=pos,
            body_ids=ids,
            is_global=True,
        )
        self._bed_grid, self._fill_kg = rft.excavate(
            self._bed_grid, self._fill_kg, self._cells, centre, rft.drum_axis(quat[:, 2:]),
            self._capacity, self._rft,
        )

    def _soil(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bed_grid, self._fill_kg

    def _relay_soil(self, env_ids: torch.Tensor) -> None:
        self._bed_grid[env_ids] = self._fresh_bed
