# Terrain scans and procedural terrain.
#
# A scan is (num_envs, ny * nx) heights in the machine's yaw frame, measured
# as reference_z - ground_z (chassis height minus ground height), clipped.

from __future__ import annotations

import numpy as np
import torch

from .course import COURSE_LAYOUTS
from .observations import (
    DIG_CRITIC_FAR_BIAS,
    DIG_CRITIC_FAR_CELL,
    DIG_CRITIC_FAR_NX,
    DIG_CRITIC_FAR_NY,
    DIG_CRITIC_NEAR_BIAS,
    DIG_CRITIC_NEAR_CELL,
    DIG_CRITIC_NEAR_NX,
    DIG_CRITIC_NEAR_NY,
    DIG_SCAN_CELL,
    DIG_SCAN_NX,
    DIG_SCAN_NY,
    NAV_NEAR_CELL,
    NAV_NEAR_NX,
    NAV_NEAR_NY,
)


# --- scan patterns ---


def scan_pattern(nx: int, ny: int, cell: float, forward_bias: float = 0.0, device="cpu") -> torch.Tensor:
    """(ny * nx, 2) local xy offsets of a grid centred forward_bias ahead of
    the origin, row-major in (y, x)."""
    xs = (torch.arange(nx, device=device, dtype=torch.float32) - (nx - 1) / 2) * cell + forward_bias
    ys = (torch.arange(ny, device=device, dtype=torch.float32) - (ny - 1) / 2) * cell
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def nav_near_pattern(device="cpu") -> torch.Tensor:
    return scan_pattern(NAV_NEAR_NX, NAV_NEAR_NY, NAV_NEAR_CELL, device=device)


def dig_critic_far_pattern(device="cpu") -> torch.Tensor:
    return scan_pattern(DIG_CRITIC_FAR_NX, DIG_CRITIC_FAR_NY, DIG_CRITIC_FAR_CELL, DIG_CRITIC_FAR_BIAS, device)


def dig_critic_near_pattern(device="cpu") -> torch.Tensor:
    return scan_pattern(DIG_CRITIC_NEAR_NX, DIG_CRITIC_NEAR_NY, DIG_CRITIC_NEAR_CELL, DIG_CRITIC_NEAR_BIAS, device)


def dig_scan_pattern(device="cpu") -> torch.Tensor:
    return scan_pattern(DIG_SCAN_NX, DIG_SCAN_NY, DIG_SCAN_CELL, device=device)


def degrade_scan(
    scan: torch.Tensor,
    pattern: torch.Tensor,
    noise_std: float,
    dropout: float,
    range_ref: float,
    invalid_value: float,
) -> torch.Tensor:
    """Depth-sensor model on an (E, N) scan taken at the (N, 2) pattern:
    Gaussian noise and dropout, both scaled by 1 + range / range_ref. A
    dropped cell reads invalid_value."""
    scale = 1.0 + torch.linalg.norm(pattern, dim=-1).unsqueeze(0) / range_ref
    out = scan + torch.randn_like(scan) * (noise_std * scale)
    miss = torch.rand_like(scan) < (dropout * scale).clamp(0.0, 1.0)
    return torch.where(miss, torch.full_like(out, invalid_value), out)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw of (E, 4) quaternions in (x, y, z, w) order."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def scan_points_world(
    origin_w: torch.Tensor,
    quat_w: torch.Tensor,
    pattern: torch.Tensor,
    yaw_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """(E, N, 2) world xy of the pattern placed at origin_w (E, 3), rotated
    by the yaw of quat_w plus yaw_offset (E,)."""
    yaw = yaw_from_quat(quat_w)
    if yaw_offset is not None:
        yaw = yaw + yaw_offset
    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    px, py = pattern[:, 0].unsqueeze(0), pattern[:, 1].unsqueeze(0)
    return torch.stack([origin_w[:, 0:1] + c * px - s * py, origin_w[:, 1:2] + s * px + c * py], dim=-1)


def sample_height_grid(
    grid: torch.Tensor,
    query_xy: torch.Tensor,
    env_origins: torch.Tensor,
    grid_lower: tuple[float, float],
    cell_size: float,
    outside_value: float,
) -> torch.Tensor:
    """Bilinear sample (E, N) of per-env (E, ny, nx) height grids, env-local
    with cell centres from grid_lower, at (E, N, 2) world xy. outside_value off
    the grid."""
    E, ny, nx = grid.shape
    local = query_xy - env_origins[:, None, :2]
    fx = (local[..., 0] - grid_lower[0]) / cell_size - 0.5
    fy = (local[..., 1] - grid_lower[1]) / cell_size - 0.5
    x0, y0 = torch.floor(fx), torch.floor(fy)
    tx, ty = fx - x0, fy - y0
    x0, y0 = x0.long(), y0.long()
    inside = (x0 >= 0) & (x0 + 1 < nx) & (y0 >= 0) & (y0 + 1 < ny)
    xc, yc = x0.clamp(0, nx - 2), y0.clamp(0, ny - 2)
    flat = grid.reshape(E, ny * nx)

    def at(iy, ix):
        return torch.gather(flat, 1, iy * nx + ix)

    top = at(yc, xc) * (1 - tx) + at(yc, xc + 1) * tx
    bot = at(yc + 1, xc) * (1 - tx) + at(yc + 1, xc + 1) * tx
    out = top * (1 - ty) + bot * ty
    return torch.where(inside, out, torch.full_like(out, outside_value))


def scan_from_heightfield(
    heightfield: torch.Tensor,
    origin_w: torch.Tensor,
    quat_w: torch.Tensor,
    pattern: torch.Tensor,
    env_origins: torch.Tensor,
    grid_lower: tuple[float, float],
    cell_size: float,
    reference_z: torch.Tensor,
    clip: float = 1.0,
    outside_value: float = -1.0,
    yaw_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scan of per-env height grids: reference_z (E,) minus the ground under
    the placed pattern, clipped to +-clip with NaN read as clip."""
    pts = scan_points_world(origin_w, quat_w, pattern, yaw_offset)
    h = sample_height_grid(heightfield, pts, env_origins, grid_lower, cell_size, outside_value)
    out = reference_z.unsqueeze(-1) - (h + env_origins[:, 2:3])
    return torch.nan_to_num(out, nan=clip, posinf=clip, neginf=-clip).clamp(-clip, clip)


def scan_from_raycaster(chassis_pos_w: torch.Tensor, ray_hits_w: torch.Tensor, clip: float = 1.0) -> torch.Tensor:
    """Scan from a RayCaster: chassis z (E, 3) minus each ray hit's z (E, N, 3),
    clipped to +-clip. A missed ray reads clip."""
    h = chassis_pos_w[:, 2:3] - ray_hits_w[..., 2]
    return torch.nan_to_num(h, nan=clip, posinf=clip, neginf=-clip).clamp(-clip, clip)


# --- procedural terrain (numpy, for Isaac Lab's height-field terrain generator) ---


def excavation_height_field_np(
    width_pixels: int,
    length_pixels: int,
    horizontal_scale: float,
    vertical_scale: float,
    rng: np.random.Generator,
    difficulty: float = 1.0,
    num_pits: int = 3,
    num_piles: int = 3,
    pit_depth: tuple[float, float] = (0.05, 0.22),
    pile_height: tuple[float, float] = (0.05, 0.20),
    feature_radius: tuple[float, float] = (0.5, 1.2),
    slope: float = 0.04,
    noise: float = 0.01,
) -> np.ndarray:
    """Worked ground as an int16 (width_pixels, length_pixels) height field in
    units of vertical_scale, [i, j] at (i, j) * horizontal_scale: a tilt,
    Gaussian pits and piles, and uniform noise. Amplitudes scale by
    0.35 + 0.65 * difficulty."""
    xs = np.arange(width_pixels, dtype=np.float64) * horizontal_scale
    ys = np.arange(length_pixels, dtype=np.float64) * horizontal_scale
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    cx, cy = xs.mean(), ys.mean()
    amp = 0.35 + 0.65 * difficulty

    tilt = rng.uniform(-slope, slope, size=2) * amp
    field = (gx - cx) * tilt[0] + (gy - cy) * tilt[1]
    for count, (lo, hi), sign in ((num_pits, pit_depth, -1.0), (num_piles, pile_height, 1.0)):
        for _ in range(count):
            fx = rng.uniform(xs.min(), xs.max())
            fy = rng.uniform(ys.min(), ys.max())
            rad = rng.uniform(*feature_radius)
            a = rng.uniform(lo, hi) * amp
            field += sign * a * np.exp(-((gx - fx) ** 2 + (gy - fy) ** 2) / (2.0 * rad ** 2))
    field += rng.uniform(-noise, noise, size=field.shape)
    return np.rint(field / vertical_scale).astype(np.int16)


def hazard_relief_np(
    width_pixels: int,
    length_pixels: int,
    horizontal_scale: float,
    rng: np.random.Generator,
    difficulty: float = 1.0,
    num_craters: int = 6,
    crater_radius: tuple[float, float] = (1.0, 1.8),
    crater_depth: tuple[float, float] = (0.40, 0.70),
    crater_wall_power: float = 16.0,
    rim_height: tuple[float, float] = (0.10, 0.20),
    rim_width: float = 0.5,
    num_boulders: int = 12,
    boulder_radius: tuple[float, float] = (0.40, 0.70),
    boulder_height: tuple[float, float] = (0.35, 0.60),
    clear_radius: float = 6.5,
    hazard_extent: float = 14.0,
    min_gap: float = 1.8,
    max_attempts: int = 200,
) -> np.ndarray:
    """Craters and boulders in metres on excavation_height_field_np's grid,
    to add to a base field.

    round(difficulty * num_craters) craters, then round(difficulty *
    num_boulders) boulders, centred within hazard_extent of the cell centre,
    none reaching within clear_radius of it, each pair min_gap apart edge to
    edge. A hazard with no free spot in max_attempts draws is skipped.
    """
    xs = np.arange(width_pixels, dtype=np.float64) * horizontal_scale
    ys = np.arange(length_pixels, dtype=np.float64) * horizontal_scale
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    cx, cy = xs.mean(), ys.mean()
    relief = np.zeros((width_pixels, length_pixels), dtype=np.float64)
    placed: list[tuple[float, float, float]] = []    # (x, y, footprint radius)

    def place(footprint: float) -> tuple[float, float] | None:
        for _ in range(max_attempts):
            x = rng.uniform(cx - hazard_extent, cx + hazard_extent)
            y = rng.uniform(cy - hazard_extent, cy + hazard_extent)
            if np.hypot(x - cx, y - cy) < clear_radius + footprint:
                continue
            if any(np.hypot(x - px, y - py) < footprint + pr + min_gap for px, py, pr in placed):
                continue
            placed.append((x, y, footprint))
            return x, y
        return None

    for _ in range(int(round(difficulty * num_craters))):
        radius = rng.uniform(*crater_radius)
        depth = rng.uniform(*crater_depth)
        rim = rng.uniform(*rim_height)
        at = place(radius + rim_width)
        if at is not None:
            relief += crater_relief(gx, gy, *at, radius, depth, rim, rim_width, crater_wall_power)

    for _ in range(int(round(difficulty * num_boulders))):
        radius = rng.uniform(*boulder_radius)
        height = rng.uniform(*boulder_height)
        at = place(radius)
        if at is not None:
            relief += boulder_relief(gx, gy, *at, radius, height)
    return relief


def crater_relief(gx, gy, x, y, radius, depth, rim, rim_width, wall_power) -> np.ndarray:
    """A crater at (x, y): a bowl -depth * (1 - (r / radius)^wall_power)
    inside radius, and a rim falling from rim to 0 over rim_width outside it."""
    r = np.hypot(gx - x, gy - y)
    bowl = -depth * (1.0 - (r / radius) ** wall_power)
    lip = rim * (1.0 - (r - radius) / rim_width)
    return np.where(r < radius, bowl, np.where(r < radius + rim_width, lip, 0.0))


def boulder_relief(gx, gy, x, y, radius, height) -> np.ndarray:
    """A flat-topped boulder of height at (x, y)."""
    return np.where(np.hypot(gx - x, gy - y) < radius, height, 0.0)


def course_relief_np(
    width_pixels: int,
    length_pixels: int,
    horizontal_scale: float,
    layout: str,
    crater_wall_power: float = 16.0,
    rim_width: float = 0.5,
) -> np.ndarray:
    """One COURSE_LAYOUTS layout in metres on hazard_relief_np's grid,
    obstacles placed from the cell centre."""
    xs = np.arange(width_pixels, dtype=np.float64) * horizontal_scale
    ys = np.arange(length_pixels, dtype=np.float64) * horizontal_scale
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    cx, cy = xs.mean(), ys.mean()
    relief = np.zeros((width_pixels, length_pixels), dtype=np.float64)
    for kind, x, y, radius, *size in COURSE_LAYOUTS[layout]:
        if kind == "crater":
            depth, rim = size
            relief += crater_relief(gx, gy, cx + x, cy + y, radius, depth, rim, rim_width, crater_wall_power)
        else:
            relief += boulder_relief(gx, gy, cx + x, cy + y, radius, size[0])
    return relief
