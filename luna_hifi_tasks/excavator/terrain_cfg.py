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

from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass

from .mdp.terrain import excavation_height_field_np, hazard_relief_np

# Name of the flat-patch set the navigate env draws its goals from.
GOAL_PATCHES = "goal"


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


@height_field_to_mesh
def hazard_terrain(
    difficulty: float, cfg: "HfHazardTerrainCfg"
) -> tuple[np.ndarray, np.ndarray]:
    """Gentle worked ground with craters and boulders on it; see
    mdp/terrain.hazard_relief_np. Same contract as excavation_terrain.

    The hazards stay clear of the cell centre, so _terrain_origin's tallest
    point near the centre is the base ground's.
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    base = excavation_height_field_np(
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
    relief = hazard_relief_np(
        width_pixels,
        length_pixels,
        cfg.horizontal_scale,
        rng,
        difficulty=difficulty,
        num_craters=cfg.num_craters,
        crater_radius=cfg.crater_radius,
        crater_depth=cfg.crater_depth,
        crater_wall_power=cfg.crater_wall_power,
        rim_height=cfg.rim_height,
        rim_width=cfg.rim_width,
        num_boulders=cfg.num_boulders,
        boulder_radius=cfg.boulder_radius,
        boulder_height=cfg.boulder_height,
        clear_radius=cfg.clear_radius,
        hazard_extent=cfg.hazard_extent,
        min_gap=cfg.min_gap,
    )
    field = np.rint(base + relief / cfg.vertical_scale).astype(np.int16)
    return field, _terrain_origin(field, cfg)


@configclass
class HfHazardTerrainCfg(HfExcavationTerrainCfg):
    """Ground the machine has to steer around: craters it cannot climb out of
    and boulders it cannot climb over, on a base gentle enough that a flat
    patch test tells the two apart (see GOAL_PATCH_HAZARD_DIFF).

    The base fields are HfExcavationTerrainCfg's, overridden gentle: at most
    0.11 m of tilt across a 4 m disc and 0.05 m pits and piles, so a 2 m
    radius around any point spans about 0.22 m at worst.

    clear_radius covers where the env spawns the machine: spawn_xy_jitter of
    3 m reaches 4.24 m out on the diagonal, plus 1.8 m of machine
    half-length, plus margin.
    """

    function = hazard_terrain

    num_pits: int = 2
    num_piles: int = 2
    pit_depth: tuple[float, float] = (0.02, 0.05)
    pile_height: tuple[float, float] = (0.02, 0.05)
    slope: float = 0.02
    noise: float = 0.005

    num_craters: int = 6                 # at difficulty 1
    crater_radius: tuple[float, float] = (1.0, 1.8)
    crater_depth: tuple[float, float] = (0.40, 0.70)
    crater_wall_power: float = 16.0
    rim_height: tuple[float, float] = (0.10, 0.20)
    rim_width: float = 0.5
    num_boulders: int = 12               # at difficulty 1
    boulder_radius: tuple[float, float] = (0.40, 0.70)
    boulder_height: tuple[float, float] = (0.35, 0.60)
    clear_radius: float = 6.5
    hazard_extent: float = 14.0          # hazard centres within this of the cell centre
    min_gap: float = 1.8                 # edge to edge, more than the 1.35 m machine width


# Goal spots for the navigate env, found per cell at startup by Isaac Lab's
# flat-patch sampler and drawn from by ExcavatorNavigateEnv._sample_goals.
# They span what goals did before, the cell less a 3 m margin: +-13 m of a
# 32 m cell. 512 per cell puts a spot every ~1.15 m on average.
#
# A spot is tested by ray-casting rings around it, nine directions 40 deg
# apart at each radius (the sampler's linspace repeats the first as a tenth),
# and rejected if the hits span more than max_height_diff. 2.0 m covers the
# machine at any heading. Rings that sparse can miss the edge of a hazard
# between two hits; on generated hazard cells no accepted spot had a crater
# or boulder closer than 0.68 m, and 0.35% had one within 1.0 m, about where
# the wheels reach.
_GOAL_PATCH_RADII = [0.0, 0.5, 1.0, 1.5, 2.0]
_GOAL_PATCH_COUNT = 512
_GOAL_PATCH_HALF = 13.0


def _goal_patches(max_height_diff: float) -> FlatPatchSamplingCfg:
    """A fresh cfg per sub-terrain, so an override on one cannot reach another."""
    return FlatPatchSamplingCfg(
        num_patches=_GOAL_PATCH_COUNT,
        patch_radius=list(_GOAL_PATCH_RADII),
        x_range=(-_GOAL_PATCH_HALF, _GOAL_PATCH_HALF),
        y_range=(-_GOAL_PATCH_HALF, _GOAL_PATCH_HALF),
        max_height_diff=max_height_diff,
    )


# Worked cells reject nothing, so their goals are spread as they always were.
GOAL_PATCH_ANY_DIFF = 100.0
# Hazard cells: the base ground spans 0.22 m at worst within 2 m, a boulder
# stands at least 0.35 m and a crater sinks at least 0.40 m.
GOAL_PATCH_HAZARD_DIFF = 0.25


# Four sub-terrain types, so a single row mixes easy and hard cells. Cells
# are 32 m against a 3.4 m machine and goal distances up to 18 m. Ten rows is
# the difficulty ladder and twelve columns the variations at each rung: 120
# cells, about 8 envs per cell at 1024. The generator gives each type whole
# columns in proportion, here 2 / 2 / 2 / 6, and an env keeps its column for
# the run while its row moves with the curriculum, so half the envs drive
# among hazards whose count grows with their level.
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
            proportion=2 / 12, num_pits=2, num_piles=2,
            pit_depth=(0.03, 0.12), pile_height=(0.03, 0.10), noise=0.005,
            flat_patch_sampling={GOAL_PATCHES: _goal_patches(GOAL_PATCH_ANY_DIFF)},
        ),
        "worked": HfExcavationTerrainCfg(
            proportion=2 / 12,
            flat_patch_sampling={GOAL_PATCHES: _goal_patches(GOAL_PATCH_ANY_DIFF)},
        ),
        "worked_heavy": HfExcavationTerrainCfg(
            proportion=2 / 12, num_pits=5, num_piles=5,
            pit_depth=(0.10, 0.22), pile_height=(0.10, 0.20), slope=0.06,
            flat_patch_sampling={GOAL_PATCHES: _goal_patches(GOAL_PATCH_ANY_DIFF)},
        ),
        "hazards": HfHazardTerrainCfg(
            proportion=6 / 12,
            flat_patch_sampling={GOAL_PATCHES: _goal_patches(GOAL_PATCH_HAZARD_DIFF)},
        ),
    },
)
