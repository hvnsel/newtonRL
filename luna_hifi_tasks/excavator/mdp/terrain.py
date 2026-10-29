# terrain.py
#
# One terrain-scan interface, two backends.
#
#   heightfield  -- bilinear sample of a static height grid. The rigid tier:
#                   procedurally generated excavation-like terrain, thousands
#                   of envs, no MPM. This is where navigation trains.
#   particles    -- rasterise live MPM particles, then sample the same way.
#                   The deformable tier, where excavation trains.
#
# Both return the same tensor: (num_envs, ny*nx) of chassis-relative heights
# in the robot's yaw frame, as reference_z minus ground_z, so a lower surface
# reads larger. A skill trained on one backend runs unmodified on the other.

from __future__ import annotations

import numpy as np
import torch

from .course import COURSE_LAYOUTS
from .observations import (
    DIG_SCAN_CELL,
    DIG_SCAN_NX,
    DIG_SCAN_NY,
    NAV_FAR_BIAS,
    NAV_FAR_CELL,
    NAV_FAR_NX,
    NAV_FAR_NY,
    NAV_NEAR_BIAS,
    NAV_NEAR_CELL,
    NAV_NEAR_NX,
    NAV_NEAR_NY,
)


# ---------------------------------------------------------------------------
# Scan pattern
# ---------------------------------------------------------------------------


def scan_pattern(
    nx: int,
    ny: int,
    cell: float,
    forward_bias: float = 0.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Local (x, y) offsets of the scan grid. Returns (ny*nx, 2).

    Row-major in (y, x), so a reshape to (ny, nx) is a picture of the ground
    with +x to the right.
    """
    xs = (torch.arange(nx, device=device, dtype=torch.float32) - (nx - 1) / 2) * cell + forward_bias
    ys = (torch.arange(ny, device=device, dtype=torch.float32) - (ny - 1) / 2) * cell
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def nav_far_pattern(device="cpu") -> torch.Tensor:
    return scan_pattern(NAV_FAR_NX, NAV_FAR_NY, NAV_FAR_CELL, NAV_FAR_BIAS, device)


def nav_near_pattern(device="cpu") -> torch.Tensor:
    return scan_pattern(NAV_NEAR_NX, NAV_NEAR_NY, NAV_NEAR_CELL, NAV_NEAR_BIAS, device)


def degrade_scan(
    scan: torch.Tensor,           # (E, N) heights
    pattern: torch.Tensor,        # (N, 2) local offsets the scan was taken at
    noise_std: float,
    dropout: float,
    range_ref: float,
    invalid_value: float,
) -> torch.Tensor:
    """Model a depth sensor: noise and dropout that grow with range.

    Both terms scale with 1 + r/range_ref. A dropped cell reads
    `invalid_value`, which sits outside the clipped range of a real height.
    """
    if noise_std <= 0.0 and dropout <= 0.0:
        return scan
    scale = 1.0 + torch.linalg.norm(pattern, dim=-1).unsqueeze(0) / range_ref
    out = scan + torch.randn_like(scan) * (noise_std * scale)
    if dropout > 0.0:
        miss = torch.rand_like(scan) < (dropout * scale).clamp(0.0, 1.0)
        out = torch.where(miss, torch.full_like(out, invalid_value), out)
    return out


def dig_scan_pattern(device="cpu") -> torch.Tensor:
    return scan_pattern(DIG_SCAN_NX, DIG_SCAN_NY, DIG_SCAN_CELL, 0.0, device)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw only, from an (x, y, z, w) quaternion. Returns (E,).

    The order is Isaac Lab 3.x's xyzw.
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def scan_points_world(
    origin_w: torch.Tensor,     # (E, 3) chassis or drum position
    quat_w: torch.Tensor,       # (E, 4) x,y,z,w
    pattern: torch.Tensor,      # (N, 2)
    yaw_offset: torch.Tensor | None = None,   # (E,) rad, added to the chassis yaw
) -> torch.Tensor:
    """Place the scan pattern in the world. Returns (E, N, 2) of xy.

    Rotated by yaw only, so the sampling grid stays flat as the machine
    pitches and rolls and the same terrain always reads the same.

    `yaw_offset` turns the pattern away from the chassis heading. The
    excavator carries a drum at each end and cuts with whichever one leads,
    so a scan taken at the rear drum has to look out the back: pi there
    points the window at unbroken ground instead of back through the machine.
    """
    yaw = yaw_from_quat(quat_w)
    if yaw_offset is not None:
        yaw = yaw + yaw_offset
    c, s = torch.cos(yaw), torch.sin(yaw)
    px, py = pattern[:, 0].unsqueeze(0), pattern[:, 1].unsqueeze(0)
    wx = origin_w[:, 0:1] + c.unsqueeze(-1) * px - s.unsqueeze(-1) * py
    wy = origin_w[:, 1:2] + s.unsqueeze(-1) * px + c.unsqueeze(-1) * py
    return torch.stack([wx, wy], dim=-1)


# ---------------------------------------------------------------------------
# Sampling a height grid
# ---------------------------------------------------------------------------


def sample_height_grid(
    grid: torch.Tensor,             # (E, ny, nx) heights in env-local frame
    query_xy: torch.Tensor,         # (E, N, 2) world xy
    env_origins: torch.Tensor,      # (E, 3)
    grid_lower: tuple[float, float],
    cell_size: float,
    outside_value: float,
) -> torch.Tensor:
    """Bilinear sample of a per-env height grid. Returns (E, N).

    Query points outside the grid return `outside_value`.
    """
    E, ny, nx = grid.shape
    local = query_xy - env_origins[:, None, :2]
    fx = (local[..., 0] - grid_lower[0]) / cell_size - 0.5
    fy = (local[..., 1] - grid_lower[1]) / cell_size - 0.5

    x0 = torch.floor(fx); y0 = torch.floor(fy)
    tx = (fx - x0).unsqueeze(-1); ty = (fy - y0).unsqueeze(-1)
    x0 = x0.long(); y0 = y0.long()

    inside = (x0 >= 0) & (x0 + 1 < nx) & (y0 >= 0) & (y0 + 1 < ny)
    xc = x0.clamp(0, nx - 2); yc = y0.clamp(0, ny - 2)

    flat = grid.reshape(E, ny * nx)
    def gather(iy, ix):
        return torch.gather(flat, 1, (iy * nx + ix).clamp(0, ny * nx - 1))

    h00 = gather(yc, xc); h10 = gather(yc, xc + 1)
    h01 = gather(yc + 1, xc); h11 = gather(yc + 1, xc + 1)
    tx = tx.squeeze(-1); ty = ty.squeeze(-1)
    top = h00 * (1 - tx) + h10 * tx
    bot = h01 * (1 - tx) + h11 * tx
    out = top * (1 - ty) + bot * ty
    return torch.where(inside, out, torch.full_like(out, outside_value))


# ---------------------------------------------------------------------------
# The two backends
# ---------------------------------------------------------------------------


def scan_from_heightfield(
    heightfield: torch.Tensor,      # (E, ny, nx) static terrain, env-local
    origin_w: torch.Tensor,
    quat_w: torch.Tensor,
    pattern: torch.Tensor,
    env_origins: torch.Tensor,
    grid_lower: tuple[float, float],
    cell_size: float,
    reference_z: torch.Tensor,      # (E,) usually the chassis height
    clip: float = 1.0,
    outside_value: float = -1.0,
    yaw_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rigid tier. Returns (E, N) chassis-relative heights, clipped."""
    pts = scan_points_world(origin_w, quat_w, pattern, yaw_offset)
    h = sample_height_grid(heightfield, pts, env_origins, grid_lower, cell_size, outside_value)
    out = reference_z.unsqueeze(-1) - (h + env_origins[:, 2:3])
    # clamp passes NaN through, so the same guard scan_from_raycaster applies:
    # a single nonfinite particle height rasterises into a cell and every
    # sample that gathers it carries the NaN into the observation and reward.
    return torch.nan_to_num(out, nan=clip, posinf=clip, neginf=-clip).clamp(-clip, clip)


# ---------------------------------------------------------------------------
# The ray-caster backend (rigid tier, in the live sim)
# ---------------------------------------------------------------------------


def scan_from_raycaster(
    sensor_pos_w: torch.Tensor,     # (E, 3)  RayCasterData.pos_w
    ray_hits_w: torch.Tensor,       # (E, N, 3)  RayCasterData.ray_hits_w
    sensor_height_offset: float,    # the z of RayCasterCfg.offset.pos
    clip: float = 1.0,
) -> torch.Tensor:
    """Rigid tier, live. Same output as the other two backends.

    Isaac Lab's height_scan observation is sensor_z - hit_z - offset, which
    with the sensor mounted `sensor_height_offset` above the chassis is
    chassis_z - ground_z. A ray that hits nothing returns an infinite hit,
    which the clip absorbs.
    """
    h = sensor_pos_w[:, 2:3] - ray_hits_w[..., 2] - sensor_height_offset
    return torch.nan_to_num(h, nan=clip, posinf=clip, neginf=-clip).clamp(-clip, clip)


# ---------------------------------------------------------------------------
# Procedural excavation-like terrain for the rigid tier
# ---------------------------------------------------------------------------


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
    """One sub-terrain, in the form Isaac Lab's height_field_to_mesh takes: an
    int16 array of shape (width_pixels, length_pixels) in units of
    vertical_scale, with index [i, j] at x = i*horizontal_scale,
    y = j*horizontal_scale.

    generate_excavation_terrain's numpy twin, for the terrain generator, which
    builds meshes on the CPU once at startup. Feature amplitudes scale with
    `difficulty`, which is what the importer's curriculum rows vary.
    """
    xs = np.arange(width_pixels, dtype=np.float64) * horizontal_scale
    ys = np.arange(length_pixels, dtype=np.float64) * horizontal_scale
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    cx, cy = xs.mean(), ys.mean()

    amp = 0.35 + 0.65 * float(np.clip(difficulty, 0.0, 1.0))
    field = np.zeros((width_pixels, length_pixels), dtype=np.float64)

    tilt = rng.uniform(-slope, slope, size=2) * amp
    field += (gx - cx) * tilt[0] + (gy - cy) * tilt[1]

    for count, (lo, hi), sign in ((num_pits, pit_depth, -1.0), (num_piles, pile_height, 1.0)):
        for _ in range(count):
            fx = rng.uniform(xs.min(), xs.max())
            fy = rng.uniform(ys.min(), ys.max())
            rad = rng.uniform(*feature_radius)
            a = rng.uniform(lo, hi) * amp
            d2 = (gx - fx) ** 2 + (gy - fy) ** 2
            field += sign * a * np.exp(-d2 / (2.0 * rad ** 2))

    if noise > 0:
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
    """Craters and boulders the machine cannot drive through, in metres, on
    the same (width_pixels, length_pixels) grid as excavation_height_field_np.
    Added to a base field by the caller.

    The machine stands on 0.30 m wheels with 0.025 m cleats and 0.16 m of
    ground clearance under the deck, and the mesh turns any flank steeper than
    the generator's slope_threshold into a vertical wall:

      crater   a bowl of depth D whose wall rises as (r/R)^crater_wall_power,
               so the floor is near flat and the top of the wall is vertical,
               behind a rim of height H that falls away outward over
               rim_width. On generated cells the rim step plus the vertical
               part of the wall measures at least 0.38 m, more than the
               0.325 m a wheel reaches with its cleats.
      boulder  a flat-topped block whose whole height, 0.35-0.60 m, is a
               vertical edge: more than a wheel reaches and more than the
               deck clears.

    Counts scale with `difficulty`, which the generator ties to the curriculum
    row; the sizes do not, so any hazard present is a real one. None sits
    within clear_radius of the cell centre, where the env spawns the machine,
    and every pair stands min_gap apart edge to edge, so a route between any
    two always exists. A hazard that finds no free spot in max_attempts draws
    is skipped rather than forced in.
    """
    xs = np.arange(width_pixels, dtype=np.float64) * horizontal_scale
    ys = np.arange(length_pixels, dtype=np.float64) * horizontal_scale
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    cx, cy = xs.mean(), ys.mean()
    d = float(np.clip(difficulty, 0.0, 1.0))
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

    # Craters first: they are the larger, so they get the free space while
    # there is most of it.
    for _ in range(int(round(d * num_craters))):
        radius = rng.uniform(*crater_radius)
        depth = rng.uniform(*crater_depth)
        rim = rng.uniform(*rim_height)
        at = place(radius + rim_width)
        if at is None:
            continue
        relief += crater_relief(gx, gy, *at, radius, depth, rim, rim_width, crater_wall_power)

    for _ in range(int(round(d * num_boulders))):
        radius = rng.uniform(*boulder_radius)
        height = rng.uniform(*boulder_height)
        at = place(radius)
        if at is None:
            continue
        relief += boulder_relief(gx, gy, *at, radius, height)

    return relief


def crater_relief(
    gx: np.ndarray, gy: np.ndarray, x: float, y: float,
    radius: float, depth: float, rim: float, rim_width: float, wall_power: float,
) -> np.ndarray:
    """One crater centred on (x, y), in metres on the (gx, gy) grid: the bowl
    and rim hazard_relief_np describes."""
    r = np.hypot(gx - x, gy - y)
    bowl = -depth * (1.0 - (r / radius) ** wall_power)
    lip = rim * (1.0 - (r - radius) / rim_width)
    return np.where(r < radius, bowl, np.where(r < radius + rim_width, lip, 0.0))


def boulder_relief(
    gx: np.ndarray, gy: np.ndarray, x: float, y: float, radius: float, height: float
) -> np.ndarray:
    """One flat-topped boulder centred on (x, y), in metres on the (gx, gy) grid."""
    return np.where(np.hypot(gx - x, gy - y) < radius, height, 0.0)


def course_relief_np(
    width_pixels: int,
    length_pixels: int,
    horizontal_scale: float,
    layout: str,
    crater_wall_power: float = 16.0,
    rim_width: float = 0.5,
) -> np.ndarray:
    """One COURSE_LAYOUTS entry, in metres, on hazard_relief_np's grid, with
    each obstacle placed from the cell centre; see mdp/course.py."""
    if layout not in COURSE_LAYOUTS:
        raise ValueError(f"unknown course layout {layout!r}; the layouts are {sorted(COURSE_LAYOUTS)}")
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


def generate_excavation_terrain(
    num_envs: int,
    nx: int,
    ny: int,
    cell: float,
    num_pits: int = 3,
    num_piles: int = 3,
    pit_depth: tuple[float, float] = (0.05, 0.22),
    pile_height: tuple[float, float] = (0.05, 0.20),
    feature_radius: tuple[float, float] = (0.5, 1.2),
    slope: float = 0.04,
    noise: float = 0.01,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Height grids of ground this machine has worked over: pits capped near
    its own 0.19 m reach, features sized near the drum's 1.0 m swath.
    """
    def rand(*shape, lo=0.0, hi=1.0):
        r = torch.rand(*shape, device=device, generator=generator)
        return r * (hi - lo) + lo

    xs = (torch.arange(nx, device=device, dtype=torch.float32) - (nx - 1) / 2) * cell
    ys = (torch.arange(ny, device=device, dtype=torch.float32) - (ny - 1) / 2) * cell
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = gx.unsqueeze(0).expand(num_envs, ny, nx)
    gy = gy.unsqueeze(0).expand(num_envs, ny, nx)

    field = torch.zeros(num_envs, ny, nx, device=device)

    # a gentle overall tilt, different per env
    tilt = rand(num_envs, 2, lo=-slope, hi=slope)
    field = field + gx * tilt[:, 0, None, None] + gy * tilt[:, 1, None, None]

    half_x, half_y = nx * cell / 2, ny * cell / 2
    for count, amp_range, sign in ((num_pits, pit_depth, -1.0), (num_piles, pile_height, 1.0)):
        for _ in range(count):
            cxx = rand(num_envs, lo=-half_x, hi=half_x)[:, None, None]
            cyy = rand(num_envs, lo=-half_y, hi=half_y)[:, None, None]
            rad = rand(num_envs, lo=feature_radius[0], hi=feature_radius[1])[:, None, None]
            amp = rand(num_envs, lo=amp_range[0], hi=amp_range[1])[:, None, None]
            d2 = (gx - cxx) ** 2 + (gy - cyy) ** 2
            field = field + sign * amp * torch.exp(-d2 / (2.0 * rad ** 2))

    if noise > 0:
        field = field + (rand(num_envs, ny, nx) - 0.5) * 2.0 * noise
    return field
