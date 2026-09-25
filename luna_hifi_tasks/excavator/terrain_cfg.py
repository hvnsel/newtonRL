# terrain_cfg.py
#
# The rigid-tier terrain: procedurally generated ground that looks like this
# machine has already worked it over, as an Isaac Lab TerrainGeneratorCfg.
#
# The generator builds every sub-terrain as a trimesh once at startup,
# TerrainImporter lays them out in a rows x cols grid and assigns environments
# to cells, the Newton backend collides with the mesh, and the RayCasters in
# the navigate scene scan it.
#
# Rows are curriculum levels: the importer starts envs low and moves them up
# as they succeed, and `difficulty` scales feature amplitude with the row.

from __future__ import annotations

import numpy as np

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass

from .mdp.terrain import excavation_height_field_np


@height_field_to_mesh
def excavation_terrain(difficulty: float, cfg: "HfExcavationTerrainCfg") -> np.ndarray:
    """Height-field function in the form Isaac Lab's decorator takes.

    By the time this runs the decorator has shrunk `cfg.size` by the border,
    so the pixel counts below are the inner region and the returned array goes
    straight into the padded buffer, as in Isaac's random_uniform_terrain.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    return excavation_height_field_np(
        width_pixels,
        length_pixels,
        cfg.horizontal_scale,
        cfg.vertical_scale,
        rng,
        difficulty=difficulty,
        num_pits=cfg.num_pits,
        num_piles=cfg.num_piles,
        pit_depth=cfg.pit_depth,
        pile_height=cfg.pile_height,
        feature_radius=cfg.feature_radius,
        slope=cfg.slope,
        noise=cfg.noise,
    )


@configclass
class HfExcavationTerrainCfg(HfTerrainBaseCfg):
    """Pits, spoil piles, a gentle tilt and surface noise, with amplitudes
    capped near what the excavator itself produces: 0.19 m of cut.
    """

    function = excavation_terrain

    num_pits: int = 3
    num_piles: int = 3
    pit_depth: tuple[float, float] = (0.05, 0.22)
    pile_height: tuple[float, float] = (0.05, 0.20)
    feature_radius: tuple[float, float] = (0.5, 1.2)
    slope: float = 0.04
    noise: float = 0.01
    seed: int | None = None


# One sub-terrain type at three severities, so a single row mixes easy and
# hard cells. Cells are 16 m against a 3.4 m machine and goal distances up to
# 8 m.
EXCAVATION_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=4.0,
    num_rows=6,
    num_cols=8,
    # Orders the rows by difficulty, which is what max_init_terrain_level and
    # the importer's level promotion index into.
    curriculum=True,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "worked_light": HfExcavationTerrainCfg(
            proportion=0.3, num_pits=2, num_piles=2,
            pit_depth=(0.03, 0.12), pile_height=(0.03, 0.10), noise=0.005,
        ),
        "worked": HfExcavationTerrainCfg(
            proportion=0.45,
        ),
        "worked_heavy": HfExcavationTerrainCfg(
            proportion=0.25, num_pits=5, num_piles=5,
            pit_depth=(0.10, 0.22), pile_height=(0.10, 0.20), slope=0.06,
        ),
    },
)
