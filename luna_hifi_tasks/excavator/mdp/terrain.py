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
# Both return an identical tensor: (num_envs, ny*nx) of CHASSIS-RELATIVE
# heights in the robot's yaw frame. That identity is the whole point. A policy
# cannot tell which backend produced its observation, so a skill trained on
# procedural terrain runs unmodified on MPM soil, and the fidelity tier becomes
# a config choice rather than a rewrite.
#
# Training navigation on procedural terrain rather than live MPM is not a
# compromise. A procedural generator samples pits, spoil piles, ruts and slopes
# across a far wider distribution than any real dig sequence would produce, at
# a hundred times the throughput. MPM earns its cost where granular flow is the
# physics being learned, which is excavation and nothing else.

from __future__ import annotations

import numpy as np
import torch

from .observations import (
    DIG_SCAN_CELL,
    DIG_SCAN_NX,
    DIG_SCAN_NY,
    NAV_SCAN_CELL,
    NAV_SCAN_FORWARD_BIAS,
    NAV_SCAN_NX,
    NAV_SCAN_NY,
)
from .sensors import soil_heightmap


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

    Row-major in (y, x) so a reshape to (ny, nx) is a picture of the ground
    with +x to the right -- which matters only for debugging, but debugging a
    terrain observation without that is miserable.
    """
    xs = (torch.arange(nx, device=device, dtype=torch.float32) - (nx - 1) / 2) * cell + forward_bias
    ys = (torch.arange(ny, device=device, dtype=torch.float32) - (ny - 1) / 2) * cell
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def nav_scan_pattern(device="cpu") -> torch.Tensor:
    return scan_pattern(NAV_SCAN_NX, NAV_SCAN_NY, NAV_SCAN_CELL, NAV_SCAN_FORWARD_BIAS, device)


def dig_scan_pattern(device="cpu") -> torch.Tensor:
    return scan_pattern(DIG_SCAN_NX, DIG_SCAN_NY, DIG_SCAN_CELL, 0.0, device)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw only, from an (x, y, z, w) quaternion. Returns (E,).

    Order matters and is Isaac Lab 3.x's xyzw, not 2.x's wxyz. A swapped order
    here does not raise -- it yields a yaw that is wrong by a rotation, and the
    terrain scan silently samples the ground somewhere the machine is not.
    """
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def scan_points_world(
    origin_w: torch.Tensor,     # (E, 3) chassis or drum position
    quat_w: torch.Tensor,       # (E, 4) x,y,z,w
    pattern: torch.Tensor,      # (N, 2)
) -> torch.Tensor:
    """Place the scan pattern in the world. Returns (E, N, 2) of xy.

    Rotated by YAW ONLY, deliberately. Using the full orientation tilts the
    sampling grid whenever the machine pitches or rolls, so the same terrain
    reads differently depending on the chassis attitude and the policy has to
    learn to undo its own suspension geometry. Every height scanner worth
    copying does it this way.
    """
    yaw = yaw_from_quat(quat_w)
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

    Query points outside the grid return `outside_value` rather than the
    clamped edge height. Clamping invents a flat plateau extending to infinity
    past the bed, which a policy will happily learn to drive onto.
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
) -> torch.Tensor:
    """Rigid tier. Returns (E, N) chassis-relative heights, clipped."""
    pts = scan_points_world(origin_w, quat_w, pattern)
    h = sample_height_grid(heightfield, pts, env_origins, grid_lower, cell_size, outside_value)
    return (reference_z.unsqueeze(-1) - (h + env_origins[:, 2:3])).clamp(-clip, clip)


def scan_from_particles(
    particle_pos_w: torch.Tensor,
    particle_env: torch.Tensor,
    env_origins: torch.Tensor,
    origin_w: torch.Tensor,
    quat_w: torch.Tensor,
    pattern: torch.Tensor,
    bed_lower: tuple[float, float],
    bed_cell: float,
    bed_nx: int,
    bed_ny: int,
    floor_height: float,
    reference_z: torch.Tensor,
    clip: float = 1.0,
    outside_value: float = -1.0,
) -> torch.Tensor:
    """Deformable tier. Same output as scan_from_heightfield.

    Two steps rather than one: rasterise particles onto a fixed, env-aligned
    bed grid, then sample that grid at the rotated query points. Rasterising
    straight into the rotated scan frame would look cheaper but bins particles
    into cells that move every step, which makes the observation jitter as the
    machine yaws even over perfectly still soil.

    `bed_cell` should be at or below the MPM voxel size. Coarser and the scan
    smooths away the very features the drum is cutting.
    """
    bed = soil_heightmap(
        particle_pos_w, particle_env, env_origins,
        bed_lower, bed_cell, bed_nx, bed_ny, floor_height,
    )
    pts = scan_points_world(origin_w, quat_w, pattern)
    h = sample_height_grid(bed, pts, env_origins, bed_lower, bed_cell, outside_value)
    return (reference_z.unsqueeze(-1) - (h + env_origins[:, 2:3])).clamp(-clip, clip)


# ---------------------------------------------------------------------------
# The ray-caster backend (rigid tier, in the live sim)
# ---------------------------------------------------------------------------


def isaac_grid_pattern_count(size: tuple[float, float], resolution: float) -> int:
    """How many rays isaaclab.sensors.patterns.grid_pattern emits for a size
    and resolution. Reproduces its arithmetic (both endpoints included) so a
    test can pin that our scan constants produce exactly NX*NY rays."""
    nx = len(torch.arange(-size[0] / 2, size[0] / 2 + 1.0e-9, resolution))
    ny = len(torch.arange(-size[1] / 2, size[1] / 2 + 1.0e-9, resolution))
    return nx * ny


def scan_from_raycaster(
    sensor_pos_w: torch.Tensor,     # (E, 3)  RayCasterData.pos_w
    ray_hits_w: torch.Tensor,       # (E, N, 3)  RayCasterData.ray_hits_w
    sensor_height_offset: float,    # the z of RayCasterCfg.offset.pos
    clip: float = 1.0,
) -> torch.Tensor:
    """Rigid tier, live. Same output as the other two backends.

    Isaac Lab's own height_scan observation is
        sensor_z - hit_z - offset
    and with the sensor mounted `sensor_height_offset` above the chassis that
    is exactly chassis_z - ground_z, the quantity scan_from_heightfield
    returns. A ray that misses everything comes back with an inf/huge hit, so
    the clip is load-bearing here, not cosmetic.
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
    """One sub-terrain, as Isaac Lab's height_field_to_mesh wants it: an int16
    array of shape (width_pixels, length_pixels) in units of vertical_scale,
    with index [i, j] at x = i*horizontal_scale, y = j*horizontal_scale.

    This is generate_excavation_terrain's numpy twin for the terrain
    generator, which builds meshes on the CPU once at startup. Feature
    amplitudes scale with `difficulty` so the importer's curriculum rows go
    from gentle to full relief.
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
    """Height grids that look like ground this machine has worked over.

    Pits are capped near the machine's own reach (0.19 m) and features sized
    near the drum's 1.0 m swath, because terrain the excavator cannot itself
    produce teaches the navigator to avoid obstacles it will never meet.
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
