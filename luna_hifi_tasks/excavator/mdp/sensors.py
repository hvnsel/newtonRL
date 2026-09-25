# sensors.py
#
# The two quantities that every excavator policy is built on:
#
#   1. soil_heightmap   -- MPM particles rasterised into a per-env height grid
#   2. drum_fill        -- mass of MPM particles inside a drum's cavity
#
# Both are pure PyTorch, one scatter_reduce pass over the particles.
#
# Imports torch only, so the file runs without a GPU and the Newton API
# surface stays in the two adapters at the bottom.

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Height map
# ---------------------------------------------------------------------------


def soil_heightmap(
    particle_pos_w: torch.Tensor,      # (P, 3) world positions of ALL particles
    particle_env: torch.Tensor,        # (P,)   which env each particle belongs to
    env_origins: torch.Tensor,         # (E, 3)
    grid_lower: tuple[float, float],   # (x, y) of cell (0,0), in env-local frame
    cell_size: float,
    nx: int,
    ny: int,
    floor_height: float = 0.0,
) -> torch.Tensor:
    """Rasterise particles into a per-env height grid. Returns (E, ny, nx).

    Each cell takes the height of the tallest particle in it. Cells with no
    particle read `floor_height`, which is the height of the hard floor under
    the bed. Particles outside the grid are dropped.
    """
    E = env_origins.shape[0]
    device = particle_pos_w.device

    local = particle_pos_w - env_origins[particle_env]
    gx = torch.floor((local[:, 0] - grid_lower[0]) / cell_size).long()
    gy = torch.floor((local[:, 1] - grid_lower[1]) / cell_size).long()

    inside = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    gx, gy = gx[inside], gy[inside]
    env = particle_env[inside]
    z = local[inside, 2]

    flat = torch.full((E * ny * nx,), floor_height, device=device, dtype=z.dtype)
    idx = (env * ny + gy) * nx + gx
    flat.scatter_reduce_(0, idx, z, reduce="amax", include_self=True)
    return flat.view(E, ny, nx)


def heightmap_to_obs(
    heightmap: torch.Tensor,           # (E, ny, nx)
    reference: torch.Tensor | float,   # (E, 1, 1) or scalar: subtract this
    clip: float = 0.5,
    scale: float = 2.0,
) -> torch.Tensor:
    """Flatten a height grid into a policy observation.

    Heights are taken relative to a reference, the chassis height or the
    undisturbed bed surface, then clipped and scaled.
    """
    rel = (heightmap - reference).clamp(-clip, clip) * scale
    return rel.flatten(start_dim=1)


# ---------------------------------------------------------------------------
# Drum fill
# ---------------------------------------------------------------------------


def drum_fill_mass(
    particle_pos_w: torch.Tensor,      # (P, 3)
    particle_env: torch.Tensor,        # (P,)
    particle_mass: torch.Tensor | float,
    drum_pos_w: torch.Tensor,          # (E, 3) drum body origin = drum axis centre
    drum_quat_w: torch.Tensor,         # (E, 4) x,y,z,w  -- Isaac Lab 3.x order
    bore_radius: float,
    half_length: float,
    num_envs: int,
) -> torch.Tensor:
    """Mass of soil inside each drum's cavity. Returns (E,).

    A particle counts when, in the drum's own frame, it is within
    `bore_radius` of the spin axis and within `half_length` along it. The test
    is in the drum frame, which pitches with the arm.

    The spin axis is the drum body's local y, as every hinge in this machine
    is, so the radial test uses the local x and z components.
    """
    rel = particle_pos_w - drum_pos_w[particle_env]
    local = quat_apply_inverse(drum_quat_w[particle_env], rel)

    radial_sq = local[:, 0] ** 2 + local[:, 2] ** 2
    inside = (radial_sq < bore_radius ** 2) & (local[:, 1].abs() < half_length)

    if isinstance(particle_mass, torch.Tensor):
        contrib = torch.where(inside, particle_mass, torch.zeros_like(particle_mass))
    else:
        contrib = inside.to(rel.dtype) * particle_mass

    out = torch.zeros(num_envs, device=rel.device, dtype=contrib.dtype)
    out.scatter_add_(0, particle_env, contrib)
    return out


def drum_fill_fraction(fill_mass: torch.Tensor, capacity_kg: float) -> torch.Tensor:
    """Normalise fill to [0, 1+]. Unclamped at the top, so a bore capacity
    estimate that is too small shows as a reading above 1.0."""
    return fill_mass / capacity_kg


# ---------------------------------------------------------------------------
# Quaternion helper
#
# Order is (x, y, z, w), which is what every pose in isaaclab.assets carries
# on Isaac Lab 3.x. Mirrors isaaclab.utils.math.quat_apply_inverse, duplicated
# so this module stays importable without isaaclab.
# ---------------------------------------------------------------------------


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the inverse of q. Quaternion is (x, y, z, w)."""
    xyz = q[:, :3]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return v - q[:, 3:4] * t + torch.cross(xyz, t, dim=-1)


# ---------------------------------------------------------------------------
# Newton adapter
# ---------------------------------------------------------------------------


def mpm_particle_state(mpm_object) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a Newton MPMObject's particles to (positions, env index).

    The isaaclab_newton surface this reads:

        MPMObject.data.particle_pos_w   ProxyArray, wp.vec3f,
                                        shape (num_instances, particles_per_object)
        MPMObject.num_instances         int
        MPMObject.particles_per_object  int

    Particles are already shaped per-environment, so the env index is a
    repeat_interleave over a fixed stride. `.torch` is a zero-copy view onto
    the warp array, so the returned tensor aliases live simulation memory and
    is valid until the next step.

    Per-particle mass comes from `mpm_grid_particle_mass` on the spawn cfg;
    MPMObjectData does not carry it.
    """
    pos = mpm_object.data.particle_pos_w.torch          # (E, P, 3)
    num_envs = mpm_object.num_instances
    per_env = mpm_object.particles_per_object
    if pos.shape[:2] != (num_envs, per_env):
        raise RuntimeError(
            f"unexpected MPM particle layout {tuple(pos.shape)}; "
            f"expected ({num_envs}, {per_env}, 3)"
        )
    env_idx = torch.arange(num_envs, device=pos.device).repeat_interleave(per_env)
    return pos.reshape(-1, 3), env_idx


def mpm_grid_particle_mass(cfg) -> float:
    """Per-particle mass [kg] for an MPMGridCfg, by the arithmetic in
    isaaclab_newton.sim.spawners.mpm.mpm. Drum fill is reported in kilograms
    against this number.

    The lattice resolution is ceil()ed per axis, so cell_volume is the true
    extent divided by that resolution rather than
    voxel_size**3 / particles_per_cell.
    """
    import math as _math

    if getattr(cfg, "mass", None) is not None:
        return float(cfg.mass)

    lower = [float(v) for v in cfg.lower]
    upper = [float(v) for v in cfg.upper]
    extent = [u - l for u, l in zip(upper, lower)]
    if any(e <= 0.0 for e in extent):
        raise ValueError(f"MPM grid upper must exceed lower; got {cfg.lower} .. {cfg.upper}")

    ppc = float(cfg.particles_per_cell)
    voxel = float(cfg.voxel_size)
    resolution = [max(int(_math.ceil(ppc * e / voxel)), 1) for e in extent]
    cell_volume = 1.0
    for e, r in zip(extent, resolution):
        cell_volume *= e / r
    return cell_volume * float(cfg.material.density)
