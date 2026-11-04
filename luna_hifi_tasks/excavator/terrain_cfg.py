# The rigid-tier terrain as Isaac Lab TerrainGeneratorCfgs: worked ground,
# craters and boulders built as one trimesh at startup, laid out in a
# rows x cols grid. Rows are curriculum levels; `difficulty` scales feature
# amplitude with the row.

from __future__ import annotations

import numpy as np

from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils.configclass import configclass

from .mdp.terrain import course_relief_np, excavation_height_field_np, hazard_relief_np

# Names of the flat-patch sets the navigate env draws goals and spawns from.
GOAL_PATCHES = "goal"
SPAWN_PATCHES = "spawn"


def _terrain_origin(height_field: np.ndarray, cfg: HfTerrainBaseCfg) -> np.ndarray:
    """The origin [m] in the height field's local frame: the centre in xy, and
    the tallest point within 1 m of the centre in z (as Isaac Lab's hf_terrains)."""
    cx = height_field.shape[0] // 2
    cy = height_field.shape[1] // 2
    radius = max(1, int(1.0 / cfg.horizontal_scale))
    height = np.max(height_field[max(0, cx - radius) : cx + radius, max(0, cy - radius) : cy + radius])
    return np.array([
        (height_field.shape[0] - 1) * cfg.horizontal_scale / 2,
        (height_field.shape[1] - 1) * cfg.horizontal_scale / 2,
        height * cfg.vertical_scale,
    ])


def _base_field(difficulty: float, cfg: "HfExcavationTerrainCfg") -> tuple[np.ndarray, np.random.Generator, int, int]:
    """The worked-ground field for cfg's inner region (the decorator has already
    taken the border off cfg.size), with the rng and the pixel counts."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    rng = np.random.default_rng(cfg.seed)
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
    return field, rng, width_pixels, length_pixels


@height_field_to_mesh
def excavation_terrain(difficulty: float, cfg: "HfExcavationTerrainCfg") -> tuple[np.ndarray, np.ndarray]:
    """Worked ground. Returns (field, origin) for height_field_to_mesh."""
    field, _, _, _ = _base_field(difficulty, cfg)
    return field, _terrain_origin(field, cfg)


@configclass
class HfExcavationTerrainCfg(HfTerrainBaseCfg):
    """Pits, spoil piles, a tilt and surface noise."""

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
def hazard_terrain(difficulty: float, cfg: "HfHazardTerrainCfg") -> tuple[np.ndarray, np.ndarray]:
    """Worked ground with craters and boulders (mdp/terrain.hazard_relief_np).
    The origin is taken from the ground without the hazards."""
    base, rng, width_pixels, length_pixels = _base_field(difficulty, cfg)
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
    return field, _terrain_origin(base, cfg)


@configclass
class HfHazardTerrainCfg(HfExcavationTerrainCfg):
    """Craters and boulders over the whole cell on gentle base ground (at most
    about 0.22 m of relief within 2 m of any point)."""

    function = hazard_terrain

    num_pits: int = 2
    num_piles: int = 2
    pit_depth: tuple[float, float] = (0.02, 0.05)
    pile_height: tuple[float, float] = (0.02, 0.05)
    slope: float = 0.02
    noise: float = 0.005

    num_craters: int = 12                # at difficulty 1
    crater_radius: tuple[float, float] = (1.0, 1.8)
    crater_depth: tuple[float, float] = (0.40, 0.70)
    crater_wall_power: float = 16.0
    rim_height: tuple[float, float] = (0.10, 0.20)
    rim_width: float = 0.5
    num_boulders: int = 24               # at difficulty 1
    boulder_radius: tuple[float, float] = (0.40, 0.70)
    boulder_height: tuple[float, float] = (0.35, 0.60)
    clear_radius: float = 0.0
    hazard_extent: float = 15.0          # hazard centres within this of the cell centre
    min_gap: float = 1.8                 # m, edge to edge


# Goal and spawn spots, found per cell at startup by Isaac Lab's flat-patch
# sampler within +-13 m of the cell centre. A spot is kept when ray hits on
# rings of the given radii around it span less than max_height_diff.
_PATCH_HALF = 13.0
_GOAL_PATCH_RADII = [0.0, 0.5, 1.0, 1.5, 2.0]
_GOAL_PATCH_COUNT = 512
_SPAWN_PATCH_RADII = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
_SPAWN_PATCH_COUNT = 256

# Worked cells accept every spot. Hazard cells reject any spot near a boulder
# (at least 0.35 m tall) or a crater (at least 0.40 m deep).
GOAL_PATCH_ANY_DIFF = 100.0
GOAL_PATCH_HAZARD_DIFF = 0.25


def _patches(max_height_diff: float) -> dict[str, FlatPatchSamplingCfg]:
    """Goal and spawn spot samplers for one sub-terrain."""
    def sampler(count: int, radii: list[float]) -> FlatPatchSamplingCfg:
        return FlatPatchSamplingCfg(
            num_patches=count,
            patch_radius=list(radii),
            x_range=(-_PATCH_HALF, _PATCH_HALF),
            y_range=(-_PATCH_HALF, _PATCH_HALF),
            max_height_diff=max_height_diff,
        )

    return {
        GOAL_PATCHES: sampler(_GOAL_PATCH_COUNT, _GOAL_PATCH_RADII),
        SPAWN_PATCHES: sampler(_SPAWN_PATCH_COUNT, _SPAWN_PATCH_RADII),
    }


# Training terrain: 10 difficulty rows x 12 columns of 32 m cells. Columns go
# to sub-terrain types in proportion (1 / 1 / 1 / 9); an env keeps its column
# while the curriculum moves its row. About 24M triangles, cached after the
# first build.
EXCAVATION_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(32.0, 32.0),
    border_width=4.0,
    num_rows=10,
    num_cols=12,
    curriculum=True,                     # rows ordered by difficulty
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "worked_light": HfExcavationTerrainCfg(
            proportion=1 / 12, num_pits=2, num_piles=2,
            pit_depth=(0.03, 0.12), pile_height=(0.03, 0.10), noise=0.005,
            flat_patch_sampling=_patches(GOAL_PATCH_ANY_DIFF),
        ),
        "worked": HfExcavationTerrainCfg(
            proportion=1 / 12,
            flat_patch_sampling=_patches(GOAL_PATCH_ANY_DIFF),
        ),
        "worked_heavy": HfExcavationTerrainCfg(
            proportion=1 / 12, num_pits=5, num_piles=5,
            pit_depth=(0.10, 0.22), pile_height=(0.10, 0.20), slope=0.06,
            flat_patch_sampling=_patches(GOAL_PATCH_ANY_DIFF),
        ),
        "hazards": HfHazardTerrainCfg(
            proportion=9 / 12,
            flat_patch_sampling=_patches(GOAL_PATCH_HAZARD_DIFF),
        ),
    },
)


@height_field_to_mesh
def course_terrain(difficulty: float, cfg: "HfCourseTerrainCfg") -> tuple[np.ndarray, np.ndarray]:
    """Level ground with one COURSE_LAYOUTS entry (mdp/course.py). The origin
    is taken from the ground without the obstacles."""
    base, _, width_pixels, length_pixels = _base_field(difficulty, cfg)
    relief = course_relief_np(
        width_pixels,
        length_pixels,
        cfg.horizontal_scale,
        cfg.layout,
        crater_wall_power=cfg.crater_wall_power,
        rim_width=cfg.rim_width,
    )
    field = np.rint(base + relief / cfg.vertical_scale).astype(np.int16)
    return field, _terrain_origin(base, cfg)


@configclass
class HfCourseTerrainCfg(HfHazardTerrainCfg):
    """Level ground and one fixed obstacle layout, for the playback course."""

    function = course_terrain

    layout: str = "boulder"
    num_pits: int = 0
    num_piles: int = 0
    slope: float = 0.0


# One course cell, for Luna-Excavator-Navigate-Course. The env reads
# terrain_levels, which needs curriculum=True, and spawn and goal patches on
# every cell.
COURSE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(32.0, 32.0),
    border_width=4.0,
    num_rows=1,
    num_cols=1,
    curriculum=True,
    difficulty_range=(1.0, 1.0),
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "course": HfCourseTerrainCfg(
            proportion=1.0,
            flat_patch_sampling=_patches(GOAL_PATCH_ANY_DIFF),
        ),
    },
)
