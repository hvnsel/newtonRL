# excavate_rft_env.py
#
# The excavate task with the analytical soil of mdp/rft.py in place of MPM.
# The bed is the height grid _bed_grid and the drums' load is _fill_kg. Each
# time the actions are applied: the soil's loads on both drums go onto the
# rotor and shroud bodies as world-frame wrenches, then the soil inside the
# vane-tip circles that the openings face moves from the grid into the drums. Observations, rewards
# and resets are the excavate env's. With soil_marker the grid is drawn in the
# viewer: as one smooth mesh in Newton's GL viewer, else one box per cell. With
# wheel_soil the wheels stand on the same soil:
# their colliders are off, rft.py carries them, and their tyres press ruts.

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F
from newton import ShapeFlags

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab_newton.physics import NewtonBuilderCfg

from ..excavator import BODY_CHASSIS, BODY_WHEELS
from ..excavator_cfg import EXCAVATOR_PRIM_REGEX, LUNAR_G
from ..excavator_env_base import ExcavatorEnvBase
from ..mdp import rft
from .excavate_env import ExcavatorExcavateEnv
from .excavate_env_cfg import BED_FLOOR_Z

# The arms and everything on them. Their colliders are off: the drums and
# shrouds meet the soil through rft.py, and only the wheels touch the ground
# (with wheel_soil, only the chassis, should it reach the bed floor).
ARM_COLLIDERS = EXCAVATOR_PRIM_REGEX + r"/Geometry/chassis/arm_(front|rear)_body/.*"
# The tyres and grousers, off with wheel_soil.
WHEEL_COLLIDERS = EXCAVATOR_PRIM_REGEX + r"/Geometry/chassis/wheel_(fl|fr|rl|rr)_body/.*"

SOIL_RGB = (0.62, 0.55, 0.45)    # as play_dig colours MPM particles
PILE_RGB = (0.80, 0.73, 0.58)    # soil laid above where the ground started
# Ground below where it started darkens from the soil's colour toward
# DEPTH_DARK, reached DEPTH_SHADES * SHADE_STEP down (on the mesh smoothly, on
# the boxes a step per SHADE_STEP; a deep site sets its own shade_depth):
# gently, so the centimetre or two a cut's floor varies hardly shows while
# 3 cm against 13 cm does. The depth it is shaded by is averaged over
# SHADE_BLUR, so pass seams and single cells do not stand out; the surface's
# heights are the soil's own. Soil PILE_RISE or more above where the ground
# started is a pile's colour.
SHADE_STEP = 0.01                # m
DEPTH_SHADES = 20
DEPTH_DARK = tuple(0.5 * c for c in SOIL_RGB)
SHADE_BLUR = 0.2                 # m
PILE_RISE = 0.025                # m


def depth_shades() -> list[tuple[float, ...]]:
    """DEPTH_SHADES colours from the soil's toward DEPTH_DARK, one per SHADE_STEP down."""
    return [tuple(s + k / DEPTH_SHADES * (d - s) for s, d in zip(SOIL_RGB, DEPTH_DARK))
            for k in range(1, DEPTH_SHADES + 1)]


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
            intake=c.drum_intake, repose=math.tan(math.radians(c.soil_repose)), compaction=c.soil_compaction,
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
        # (ny, nx) where _show_soil shades the ground by depth; None: everywhere.
        # And how deep the shade reaches DEPTH_DARK, m.
        self.shade_area: torch.Tensor | None = None
        self.shade_depth = DEPTH_SHADES * SHADE_STEP
        # Time between _apply_action calls, which dig: Isaac Lab calls it every
        # physics step unless Newton runs the whole decimation in one.
        self._dig_dt = self.physics_dt * (c.decimation if self._physics_handles_decimation else 1)
        # Front rotor, rear rotor, front shroud, rear shroud.
        self._tool_ids = self._drum_body_ids + self._shroud_body_ids
        # Wheels on the soil: left front, left rear, right front, right rear.
        self._wheel_patches = rft.wheel_patches(device=dev)
        self._wheel_body_ids = self.robot.find_bodies(BODY_WHEELS, preserve_order=True)[0] if c.wheel_soil else []
        # The chassis frame on the soil too (chassis_soil, with the wheels).
        self._chassis_ids = self.robot.find_bodies([BODY_CHASSIS])[0] if c.chassis_soil and c.wheel_soil else []
        self._frame_patches = rft.frame_patches(device=dev) if self._chassis_ids else None
        self._soil_marker = self._make_soil_marker() if c.soil_marker else None
        # The smooth mesh's colours, and its fixed parts, made on its first draw.
        self._soil_rgb = torch.tensor([SOIL_RGB, PILE_RGB, DEPTH_DARK], device=dev)
        self._soil_mesh = None
        print(
            f"[excavate] soil: rft.py, density {c.soil_density:.0f} kg/m3, friction {c.soil_friction:.2f} "
            f"-> stress scale {c.rft_scale:.2f} Kp rho g = {self._rft.zeta:.0f} N/m3, cohesion {c.soil_cohesion:.0f} Pa "
            f"-> +{self._rft.cohesion_depth:.3f} m depth; height grid {c.bed_grid_nx} x {c.bed_grid_ny} "
            f"@ {c.voxel_size:.3f} m; drums hold {c.drum_capacity_kg:.1f} kg, target load {c.target_load_kg:.1f} kg, "
            f"cut ref {c.cut_volume_ref:.4f} m3; drum intake {c.drum_intake:g} kg/s, loose soil repose {c.soil_repose:g} deg"
        )
        if c.wheel_soil:
            print(f"[excavate] wheels on the soil: {self._wheel_patches.area.numel()} patches per wheel, ruts pressed into the grid")
        if self._chassis_ids:
            print(f"[excavate] the chassis frame on the soil: {self._frame_patches.area.numel()} patches; "
                  f"it bulldozes the loose soil in its way")

    def _apply_action(self):
        super()._apply_action()
        self._step_soil()

    def _step_soil(self, drums: bool = True) -> None:
        """The soil's loads onto the drums (and wheels, and the chassis frame),
        then the soil the drums take in or shove and the frame bulldozes, its
        slump, and the ruts. Without drums (the site's machine driving, its
        drums stowed clear of the ground, as navigate trains): the wheels' and
        the frame's only."""
        d = self.robot.data
        # Front and rear rotor, front and rear shroud, then any wheels: (E, B, ...),
        # positions on the drum and wheel axes.
        tools = self._tool_ids if drums else []
        ids = tools + self._wheel_body_ids + self._chassis_ids
        pos = d.body_link_pos_w.torch[:, ids]
        quat = d.body_link_quat_w.torch[:, ids]
        lin = d.body_link_lin_vel_w.torch[:, ids]
        ang = d.body_link_ang_vel_w.torch[:, ids]
        centre = pos - self.scene.env_origins[:, None]          # env-local axis points
        firm = self._bed_grid - self._loose
        w = len(tools)                                          # the wheels' first index
        f = w + len(self._wheel_body_ids)                       # the chassis's, with chassis_soil

        forces, torques = [], []
        if drums:
            shroud_f, shroud_t, rotor_f, rotor_t = rft.drum_loads(
                self._patches, centre[:, 2:4], lin[:, 2:4], quat[:, 2:4], ang[:, 2:4], quat[:, :2], ang[:, :2],
                firm, self._fill_kg, self._capacity, self._rft,
            )
            self._soil_force = rotor_f + shroud_f                # (E, 2, 3) on each drum, for play_dig
            forces, torques = [rotor_f, shroud_f], [rotor_t, shroud_t]
        if self.cfg.wheel_soil:
            wheel_f, wheel_t = rft.wheel_loads(
                self._wheel_patches, centre[:, w:f], lin[:, w:f], quat[:, w:f], ang[:, w:f], firm, self._rft,
            )
            forces.append(wheel_f)
            torques.append(wheel_t)
        if self._chassis_ids:
            frame_f, frame_t = rft.frame_loads(
                self._frame_patches, centre[:, f], lin[:, f], quat[:, f], ang[:, f], self._bed_grid, self._rft,
            )
            forces.append(frame_f[:, None])
            torques.append(frame_t[:, None])
        if forces:
            self.robot.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=torch.cat(forces, dim=1),
                torques=torch.cat(torques, dim=1),
                positions=pos,
                body_ids=ids,
                is_global=True,
            )
        body = None                                             # where loose soil may not run: drums, the frame
        if drums:
            drum_axis = rft.drum_axis(quat[:, 2:4])
            self._bed_grid, self._loose, self._fill_kg = rft.excavate(
                self._bed_grid, self._loose, self._fill_kg, self._cells, centre[:, 2:4], drum_axis,
                rft.drum_inlet(quat[:, 2:4]), self._capacity, self._dig_dt, self._rft,
            )
            if self._rft.repose:
                body = rft.drum_body(self._cells, centre[:, 2:4], drum_axis, self._bed_grid)
        if self._chassis_ids:
            self._bed_grid, self._loose, frame = rft.frame_shove(
                self._bed_grid, self._loose, centre[:, f], quat[:, f], lin[:, f], self._rft,
            )
            if self._rft.repose and bool(frame.any()):          # only where the frame meets the soil
                body = frame if body is None else body | frame
        if body is not None:
            self._bed_grid, self._loose = rft.slump(self._bed_grid, self._loose, body, self._rft)
        if self.cfg.wheel_soil:
            pressed = rft.press(self._bed_grid, self._cells, centre[:, w:f], rft.drum_axis(quat[:, w:f]), self._rft, self._fresh_bed)
            self._loose = (self._loose - (self._bed_grid - pressed)).clamp(min=0.0)
            self._bed_grid = pressed
        # Drawn only for the frames the viewer renders: after this physics step if
        # the step counter, already advanced, is a multiple of render_interval.
        if self._soil_marker is not None and self._sim_step_counter % self.cfg.sim.render_interval == 0:
            self._show_soil()

    def _soil(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bed_grid, self._fill_kg

    def _relay_soil(self, env_ids: torch.Tensor) -> None:
        self._bed_grid[env_ids] = self._fresh_bed
        self._loose[env_ids] = 0.0

    # --- soil in the viewer ---

    def _make_soil_marker(self) -> VisualizationMarkers:
        def box(rgb):
            return sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb))

        markers = {"soil": box(SOIL_RGB), "pile": box(PILE_RGB)}
        markers.update({f"depth_{k + 1}": box(rgb) for k, rgb in enumerate(depth_shades())})
        return VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/soil", markers=markers))

    def _soil_shade(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(E, ny, nx) each cell's depth below where the ground started,
        averaged over SHADE_BLUR, in DEPTH_SHADES steps down to shade_depth, in
        shade_area if one is set (elsewhere, ruts and all, 0); and whether the
        cell is a pile's, PILE_RISE or more above."""
        depth = self._fresh_bed - self._bed_grid
        k = max(1, round(SHADE_BLUR / self.cfg.voxel_size)) | 1
        soft = F.avg_pool2d(depth.unsqueeze(1), k, stride=1, padding=k // 2, count_include_pad=False).squeeze(1)
        steps = (soft * (DEPTH_SHADES / self.shade_depth)).clamp(0, DEPTH_SHADES)
        if self.shade_area is not None:
            steps = steps * self.shade_area
        return steps, depth <= -PILE_RISE

    def _show_soil(self) -> None:
        """The soil in the viewer: as one smooth mesh per env in Newton's GL
        viewer, else as a box per cell from the floor to its height."""
        steps, pile = self._soil_shade()
        viewer = self._mesh_viewer()
        if viewer is not None:
            try:
                self._show_soil_mesh(viewer, steps, pile)
                return
            except Exception as err:            # the boxes for the rest of the run
                print(f"[excavate] soil mesh failed, drawing boxes instead: {err!r}")
                self._soil_mesh = False
        # The marker's boxes: the soil's (0), a pile's (1), the k-th depth shade (1 + k).
        steps = steps.round()
        index = torch.where(steps >= 1, 1 + steps, pile.float()).long()
        top = self._bed_grid                                              # (E, ny, nx)
        o = self.scene.env_origins[:, None, None]                         # (E, 1, 1, 3)
        centre = torch.cat([self._cells + o[..., :2], (o[..., 2] + 0.5 * (top + BED_FLOOR_Z)).unsqueeze(-1)], -1)
        side = torch.full_like(top, self.cfg.voxel_size)
        self._soil_marker.visualize(
            translations=centre.reshape(-1, 3),
            scales=torch.stack([side, side, top - BED_FLOOR_Z], -1).reshape(-1, 3),
            marker_indices=index.reshape(-1).int(),
        )

    def _mesh_viewer(self):
        """Newton's GL visualizer if it is on and the soil mesh has not failed; else None."""
        if self._soil_mesh is False:
            return None
        try:
            from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer
        except ImportError:
            return None
        sim = sim_utils.SimulationContext.instance()
        return next((v for v in (getattr(sim, "visualizers", None) or []) if isinstance(v, NewtonGLVisualizer)), None)

    def _show_soil_mesh(self, viewer, steps: torch.Tensor, pile: torch.Tensor) -> None:
        """Each env's soil as one mesh through its cells' centres at the soil's
        height, lit as a smooth surface, its cells' colours a texture on it:
        the soil's toward DEPTH_DARK with steps, without stepping, and a pile's
        where not shaded. Newton's log_mesh leaves a mesh's texture off (the
        last of its material's four values): it is turned on once the viewer
        has made the mesh, and until then the mesh is the soil's colour."""
        import warp as wp

        E, ny, nx = self._bed_grid.shape
        if self._soil_mesh is None:
            wp.init()                       # a no-op once Newton has started warp
            dev = self.device
            j, i = torch.meshgrid(torch.arange(ny - 1, device=dev), torch.arange(nx - 1, device=dev), indexing="ij")
            v = (j * nx + i).reshape(-1)
            # Two triangles per square of four cell centres, anticlockwise seen from above.
            tris = torch.stack([v, v + 1, v + nx + 1, v, v + nx + 1, v + nx], -1).reshape(-1).to(torch.int32).contiguous()
            j, i = torch.meshgrid(torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij")
            # Texel (j, i) is cell (j, i): the viewer flips an image's rows on upload.
            uv = torch.stack([(i + 0.5) / nx, 1.0 - (j + 0.5) / ny], -1).reshape(-1, 2).float().contiguous()
            self._soil_mesh = {"tris": wp.from_torch(tris), "uv": wp.from_torch(uv, dtype=wp.vec2), "points": [None] * E}
        mesh = self._soil_mesh
        soil, pile_rgb, dark = self._soil_rgb
        rgb = soil + (steps / DEPTH_SHADES).unsqueeze(-1) * (dark - soil)
        rgb = torch.where((pile & (steps < 0.5)).unsqueeze(-1), pile_rgb, rgb)
        rgb = (255.0 * rgb).round().to(torch.uint8).cpu().numpy()                      # (E, ny, nx, 3)
        o = self.scene.env_origins
        made = getattr(getattr(viewer, "_viewer", None), "objects", {})      # the meshes Newton's viewer has made
        for e in range(E):
            name = f"soil_{e}"
            gl_mesh = made.get(name)
            if gl_mesh is not None:
                gl_mesh.material = (*gl_mesh.material[:3], 1.0)
            # Kept until the next draw: the viewer reads it when it renders the frame.
            mesh["points"][e] = torch.cat([self._cells + o[e, :2], (o[e, 2] + self._bed_grid[e]).unsqueeze(-1)], -1).reshape(-1, 3).contiguous()
            viewer.log_mesh(
                name, wp.from_torch(mesh["points"][e], dtype=wp.vec3), mesh["tris"], uvs=mesh["uv"], texture=rgb[e],
                color=SOIL_RGB if gl_mesh is None else (1.0, 1.0, 1.0), roughness=0.95, metallic=0.0,
            )
