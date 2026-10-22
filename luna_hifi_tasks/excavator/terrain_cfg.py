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


def _terrain_origin(
    height_field: np.ndarray, cfg: HfTerrainBaseCfg, height: int | None = None
) -> np.ndarray:
    """The origin [m] in the generated height field's local frame.

    z is the tallest point within about a metre of the centre, which is what
    puts a machine spawned there on top of the terrain instead of inside it.
    x and y are half the physical span, so they use (N - 1) where the height
    window uses `// 2`; for an odd pixel count the two differ by a fraction of
    a pixel, which is Isaac Lab's own behaviour and not worth diverging from.

    Duplicated from isaaclab.terrains.height_field.hf_terrains, which keeps it
    private, on the same grounds as mdp/sensors.quat_apply_inverse: depending
    on a private name across a package boundary is worse than eight lines.
    """
    cx = height_field.shape[0] // 2
    cy = height_field.shape[1] // 2
    if height is None:
        radius = max(1, int(1.0 / cfg.horizontal_scale))
        height = np.max(
            height_field[
                max(0, cx - radius) : cx + radius,
                max(0, cy - radius) : cy + radius,
            ]
        )
    return np.array([
        (height_field.shape[0] - 1) * cfg.horizontal_scale / 2,
        (height_field.shape[1] - 1) * cfg.horizontal_scale / 2,
        height * cfg.vertical_scale,
    ])


@height_field_to_mesh
def excavation_terrain(
    difficulty: float, cfg: "HfExcavationTerrainCfg"
) -> tuple[np.ndarray, np.ndarray]:
    """Height-field function in the form Isaac Lab's decorator takes.

    By the time this runs the decorator has shrunk `cfg.size` by the border,
    so the pixel counts below are the inner region and the returned array goes
    straight into the padded buffer, as in Isaac's random_uniform_terrain.

    Returns (field, origin), which is what height_field_to_mesh unpacks on
    Isaac Lab 3.0: it writes the field into the padded buffer and shifts the
    origin's xy by the border width. The 3.0.0-rc1 the cluster image was
    built from took the bare field instead, so returning one array there
    raises "too many values to unpack" as the decorator tries to unpack the
    array's rows.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    field = excavation_height_field_np(
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
    return field, _terrain_origin(field, cfg)


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
# hard cells. Cells are 32 m against a 3.4 m machine and goal distances up to
# 18 m. Ten rows is the difficulty ladder and twelve columns the variations at
# each rung: 120 cells, about 8 envs per cell at 1024.
#
# At a 0.1 m horizontal scale a cell is 320 x 320 px, so this is ~24M
# triangles built on the CPU at startup. use_cache keeps that to the first run.
EXCAVATION_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(32.0, 32.0),
    border_width=4.0,
    num_rows=10,
    num_cols=12,
    # Orders the rows by difficulty, which is what max_init_terrain_level and
    # the importer's level promotion index into.
    curriculum=True,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
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
