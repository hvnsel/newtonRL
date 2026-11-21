# navigate_rft_env.py
#
# Navigate with the wheels on the analytical soil of mdp/rft.py, as on the
# excavation site (excavator/site): flat ground, the tyres' colliders off, and
# each env's wheels carried by its own soil grid, with the excavate RFT task's
# soil, keeping the ruts they press for the episode. Observations, actions,
# rewards and curriculum are navigate's, so its policy fine-tunes here.

from __future__ import annotations

import re

import torch
from newton import ShapeFlags

from isaaclab_newton.physics import NewtonBuilderCfg

from ..excavate.excavate_env_cfg import ExcavatorExcavateRftEnvCfg
from ..excavate.excavate_rft_env import WHEEL_COLLIDERS
from ..excavator import BODY_WHEELS
from ..excavator_cfg import LUNAR_G
from ..mdp import rft
from .navigate_env import ExcavatorNavigateEnv
from .navigate_env_cfg import ExcavatorNavigateRftEnvCfg


class ExcavatorNavigateRftEnv(ExcavatorNavigateEnv):
    cfg: ExcavatorNavigateRftEnvCfg

    def __init__(self, cfg: ExcavatorNavigateRftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        s, h, dev = ExcavatorExcavateRftEnvCfg, cfg.soil_half, self.device
        self._rft = rft.Soil(
            density=s.soil_density, friction=s.soil_friction, gravity=LUNAR_G, cell=cfg.soil_cell,
            lower=(-h, -h), cohesion=s.soil_cohesion, scale=s.rft_scale,
        )
        n = int(round(2.0 * h / cfg.soil_cell))
        self._cells = rft.cell_centres(self._rft, n, n, device=dev)
        # Env-local soil surface; the flat ground is at the env origin's height.
        self._soil_grid = torch.zeros(self.num_envs, n, n, device=dev)
        self._wheel_patches = rft.wheel_patches(device=dev)
        self._wheel_body_ids = self.robot.find_bodies(BODY_WHEELS, preserve_order=True)[0]
        print(f"[navigate] wheels on the soil: {n} x {n} grid at {cfg.soil_cell:g} m per env, "
              f"stress scale {s.rft_scale:g}, friction {s.soil_friction:.2f}, cohesion {s.soil_cohesion:g} Pa")

    def _setup_scene(self):
        super()._setup_scene()
        # The tyres meet the soil through rft.py: clear their collide flags in
        # Newton's model builder, as excavate_rft_env does with wheel_soil.
        builder = self.sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=self.sim.cfg.physics))
        shapes = [i for i, label in enumerate(builder.shape_label) if re.fullmatch(WHEEL_COLLIDERS, label)]
        if not shapes:
            raise RuntimeError(f"no wheel shapes match {WHEEL_COLLIDERS}")
        for i in shapes:
            builder.shape_flags[i] &= ~(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)
        print(f"[navigate] collisions off for {len(shapes) // self.num_envs} wheel shapes per env")

    def _apply_action(self):
        super()._apply_action()
        d = self.robot.data
        ids = self._wheel_body_ids
        pos = d.body_link_pos_w.torch[:, ids]
        quat = d.body_link_quat_w.torch[:, ids]
        centre = pos - self.scene.env_origins[:, None]
        force, torque = rft.wheel_loads(
            self._wheel_patches, centre, d.body_link_lin_vel_w.torch[:, ids], quat,
            d.body_link_ang_vel_w.torch[:, ids], self._soil_grid, self._rft,
        )
        self.robot.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=force, torques=torque, positions=pos, body_ids=ids, is_global=True,
        )
        self._soil_grid = rft.press(self._soil_grid, self._cells, centre, rft.drum_axis(quat), self._rft)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self._soil_grid[env_ids] = 0.0
