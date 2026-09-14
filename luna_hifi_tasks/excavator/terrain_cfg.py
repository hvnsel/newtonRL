# terrain_cfg.py
#
# The rigid-tier terrain: procedurally generated ground that looks like this
# machine has already worked it over, as an Isaac Lab TerrainGeneratorCfg.
#
# This is what navigation trains on. The generator builds every sub-terrain as
# a trimesh once at startup, TerrainImporter lays them out in a rows x cols
# grid and assigns environments to cells, and the Newton backend collides with
# the mesh (the core velocity env runs this exact path under newton_mjwarp).
# The RayCaster in the navigate scene then scans that mesh. Rows are curriculum
# levels: the importer starts envs at low rows and moves them up as they
# succeed, and `difficulty` in our height-field function scales feature
# amplitude with the row.
#
# Isaac-dependent, so not unit-tested here; the numpy function it wraps is.

from __future__ import annotations

import numpy as np

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .mdp.terrain import excavation_height_field_np


@height_field_to_mesh
def excavation_terrain(difficulty: float, cfg: "HfExcavationTerrainCfg") -> np.ndarray:
    """Height-field function in the shape Isaac Lab's decorator expects.

    By the time this runs the decorator has already shrunk `cfg.size` by the
    border, so the pixel counts below are the inner region and the returned
    array drops straight into the padded buffer. Same arithmetic as Isaac's
    own random_uniform_terrain.
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
    """Pits, spoil piles, a gentle tilt and surface noise.

    Amplitudes are capped near what the excavator can itself produce (0.19 m
    of cut), because terrain it could not have made teaches the navigator to
    avoid obstacles it will never meet.
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


# One sub-terrain type at three proportions of severity, so a single row mixes
# easy and hard cells and the curriculum has something to climb. Cell size is
# generous: the machine is 3.4 m long and a navigation episode covers up to
# ~12 m of goal distance.
EXCAVATION_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(16.0, 16.0),
    border_width=4.0,
    num_rows=6,
    num_cols=8,
    # Rows are difficulty levels, which is what makes max_init_terrain_level and
    # the importer's level promotion mean anything. Without this, `difficulty`
    # is sampled at random per cell and the rows carry no ordering.
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
