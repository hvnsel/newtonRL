# excavate_rft_env.py
#
# The excavate task with the analytical soil of mdp/rft.py in place of MPM.
# The bed is the height grid _bed_grid and the drums' load is _fill_kg. Each
# time the actions are applied: the soil's loads on both drums go onto the
# rotor and shroud bodies as world-frame wrenches, then the soil inside the
# vane-tip circles moves from the grid into the drums. Observations, rewards
# and resets are the excavate env's. With soil_marker the grid is drawn in the
# viewer, one box per cell. With wheel_soil the wheels stand on the same soil:
# their colliders are off, rft.py carries them, and their tyres press ruts.

from __future__ import annotations

import math
import re

import torch
from newton import ShapeFlags

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab_newton.physics import NewtonBuilderCfg

from ..excavator import BODY_WHEELS
from ..excavator_cfg import EXCAVATOR_PRIM_REGEX, LUNAR_G
from ..excavator_env_base import ExcavatorEnvBase
from ..mdp import rft
from .excavate_env import ExcavatorExcavateEnv
from .excavate_env_cfg import BED_FLOOR_Z

# The arms and everything on them. Their colliders are off: the drums and
# shrouds meet the soil through rft.py, and only the wheels touch the ground.
ARM_COLLIDERS = EXCAVATOR_PRIM_REGEX + r"/Geometry/chassis/arm_(front|rear)_body/.*"
# The tyres and grousers, off with wheel_soil.
WHEEL_COLLIDERS = EXCAVATOR_PRIM_REGEX + r"/Geometry/chassis/wheel_(fl|fr|rl|rr)_body/.*"

SOIL_RGB = (0.62, 0.55, 0.45)    # as play_dig colours MPM particles
CUT_RGB = (0.36, 0.30, 0.24)     # cells the drums have cut into


class ExcavatorExcavateRftEnv(ExcavatorExcavateEnv):

    def _setup_scene(self):
        ExcavatorEnvBase._setup_scene(self)
        # The scene is already cloned into Newton's model builder, which becomes the
        # model at sim.reset(). Clear the arm shapes' collide flags there, as Newton's
        # USD importer does for a disabled collider, and the wheels' with wheel_soil.
        # Body masses are already set.
        builder = self.sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=self.sim.cfg.physics))
        colliders = {"arm": ARM_COLLIDERS, "wheel": WHEEL_COLLIDERS} if self.cfg.wheel_soil else {"arm": ARM_COLLIDERS}
        for name, pattern in colliders.items():
            shapes = [i for i, label in enumerate(builder.shape_label) if re.fullmatch(pattern, label)]
            if not shapes:
                raise RuntimeError(f"no {name} shapes match {pattern}")
            for i in shapes:
                builder.shape_flags[i] &= ~(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)
            print(f"[excavate] collisions off for {len(shapes) // self.num_envs} {name} shapes per env")
        if self.cfg.soil_marker:
            # The ground sits at bed_top, over the cut: hide it. _show_soil draws the soil.
            for i, label in enumerate(builder.shape_label):
                if label.startswith(self.cfg.scene.ground.prim_path):
                    builder.shape_flags[i] &= ~ShapeFlags.VISIBLE

    def _init_soil(self) -> None:
        c, dev = self.cfg, self.device
        self._rft = rft.Soil(
            density=c.soil_density, friction=c.soil_friction, gravity=LUNAR_G,
            cell=c.voxel_size, lower=c.bed_grid_lower, cohesion=c.soil_cohesion, scale=c.rft_scale,
            intake=c.drum_intake, repose=math.tan(math.radians(c.soil_repose)),
        )
        self._patches = rft.drum_patches(device=dev)
        self._cells = rft.cell_centres(self._rft, c.bed_grid_nx, c.bed_grid_ny, device=dev)
        x, y = self._cells[..., 0], self._cells[..., 1]
        on_bed = (x >= c.bed_x[0]) & (x <= c.bed_x[1]) & (y >= c.bed_y[0]) & (y <= c.bed_y[1])
        # A fresh bed: level at bed_top, the floor around it.
        self._fresh_bed = torch.where(on_bed, c.bed_top, BED_FLOOR_Z)
        self._bed_grid[:] = self._fresh_bed
        self._capacity = torch.full((self.num_envs, 2), c.drum_capacity_kg, device=dev)
        # Loose soil on the grid, part of _bed_grid; it carries no load.
        self._loose = torch.zeros_like(self._bed_grid)
        # Time between _apply_action calls, which dig: Isaac Lab calls it every
        # physics step unless Newton runs the whole decimation in one.
        self._dig_dt = self.physics_dt * (c.decimation if self._physics_handles_decimation else 1)
        # Front rotor, rear rotor, front shroud, rear shroud.
        self._tool_ids = self._drum_body_ids + self._shroud_body_ids
        # Wheels on the soil: left front, left rear, right front, right rear.
        self._wheel_patches = rft.wheel_patches(device=dev)
        self._wheel_body_ids = self.robot.find_bodies(BODY_WHEELS, preserve_order=True)[0] if c.wheel_soil else []
        self._soil_marker = self._make_soil_marker() if c.soil_marker else None
        print(
            f"[excavate] soil: rft.py, density {c.soil_density:.0f} kg/m3, friction {c.soil_friction:.2f} "
            f"-> stress scale {c.rft_scale:.2f} Kp rho g = {self._rft.zeta:.0f} N/m3, cohesion {c.soil_cohesion:.0f} Pa "
            f"-> +{self._rft.cohesion_depth:.3f} m depth; height grid {c.bed_grid_nx} x {c.bed_grid_ny} "
            f"@ {c.voxel_size:.3f} m; drums hold {c.drum_capacity_kg:.1f} kg, target load {c.target_load_kg:.1f} kg, "
            f"cut ref {c.cut_volume_ref:.4f} m3; drum intake {c.drum_intake:g} kg/s, loose soil repose {c.soil_repose:g} deg"
        )
        if c.wheel_soil:
            print(f"[excavate] wheels on the soil: {self._wheel_patches.area.numel()} patches per wheel, ruts pressed into the grid")

    def _apply_action(self):
        super()._apply_action()
        d = self.robot.data
        # Front and rear rotor, front and rear shroud, then any wheels: (E, B, ...),
        # positions on the drum and wheel axes.
        ids = self._tool_ids + self._wheel_body_ids
        pos = d.body_link_pos_w.torch[:, ids]
        quat = d.body_link_quat_w.torch[:, ids]
        lin = d.body_link_lin_vel_w.torch[:, ids]
        ang = d.body_link_ang_vel_w.torch[:, ids]
        centre = pos - self.scene.env_origins[:, None]          # env-local axis points
        firm = self._bed_grid - self._loose

        shroud_f, shroud_t, rotor_f, rotor_t = rft.drum_loads(
            self._patches, centre[:, 2:4], lin[:, 2:4], quat[:, 2:4], ang[:, 2:4], quat[:, :2], ang[:, :2],
            firm, self._fill_kg, self._capacity, self._rft,
        )
        self._soil_force = rotor_f + shroud_f                    # (E, 2, 3) on each drum, for play_dig
        forces, torques = [rotor_f, shroud_f], [rotor_t, shroud_t]
        if self.cfg.wheel_soil:
            wheel_f, wheel_t = rft.wheel_loads(
                self._wheel_patches, centre[:, 4:], lin[:, 4:], quat[:, 4:], ang[:, 4:], firm, self._rft,
            )
            forces.append(wheel_f)
            torques.append(wheel_t)
        self.robot.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=torch.cat(forces, dim=1),
            torques=torch.cat(torques, dim=1),
            positions=pos,
            body_ids=ids,
            is_global=True,
        )
        drum_axis = rft.drum_axis(quat[:, 2:4])
        self._bed_grid, self._loose, self._fill_kg = rft.excavate(
            self._bed_grid, self._loose, self._fill_kg, self._cells, centre[:, 2:4], drum_axis,
            self._capacity, self._dig_dt, self._rft,
        )
        if self._rft.repose:
            body = rft.drum_body(self._cells, centre[:, 2:4], drum_axis, self._bed_grid)
            self._bed_grid, self._loose = rft.slump(self._bed_grid, self._loose, body, self._rft)
        if self.cfg.wheel_soil:
            pressed = rft.press(self._bed_grid, self._cells, centre[:, 4:], rft.drum_axis(quat[:, 4:]))
            self._loose = (self._loose - (self._bed_grid - pressed)).clamp(min=0.0)
            self._bed_grid = pressed
        if self._soil_marker is not None:
            self._show_soil()

    def _soil(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bed_grid, self._fill_kg

    def _relay_soil(self, env_ids: torch.Tensor) -> None:
        self._bed_grid[env_ids] = self._fresh_bed
        self._loose[env_ids] = 0.0

    # --- soil in the viewer ---

    def _make_soil_marker(self) -> VisualizationMarkers:
        look = sim_utils.PreviewSurfaceCfg
        return VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/soil",
            markers={
                "soil": sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), visual_material=look(diffuse_color=SOIL_RGB)),
                "cut": sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), visual_material=look(diffuse_color=CUT_RGB)),
            },
        ))

    def _show_soil(self) -> None:
        """Every grid cell as a box from the floor to its height, darker where cut."""
        top = self._bed_grid                                              # (E, ny, nx)
        o = self.scene.env_origins[:, None, None]                         # (E, 1, 1, 3)
        centre = torch.cat([self._cells + o[..., :2], (o[..., 2] + 0.5 * (top + BED_FLOOR_Z)).unsqueeze(-1)], -1)
        side = torch.full_like(top, self.cfg.voxel_size)
        self._soil_marker.visualize(
            translations=centre.reshape(-1, 3),
            scales=torch.stack([side, side, top - BED_FLOOR_Z], -1).reshape(-1, 3),
            marker_indices=(top < self._fresh_bed - 1e-3).reshape(-1).int(),
        )
