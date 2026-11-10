# sensors.py
#
# Soil readings from MPM particles: a per-env height grid and the mass inside
# a drum's bore, plus the adapters that read a Newton MPMObject.

from __future__ import annotations

import math

import torch


def soil_heightmap(
    particle_pos_w: torch.Tensor,      # (P, 3) world positions of all particles
    particle_env: torch.Tensor,        # (P,)   env index of each particle
    env_origins: torch.Tensor,         # (E, 3)
    grid_lower: tuple[float, float],   # (x, y) of cell (0, 0), env-local
    cell_size: float,
    nx: int,
    ny: int,
    floor_height: float = 0.0,
) -> torch.Tensor:
    """(E, ny, nx) env-local height of the tallest particle in each cell.

    Empty cells read `floor_height`. Particles outside the grid or with a
    non-finite height are dropped.
    """
    E = env_origins.shape[0]
    local = particle_pos_w - env_origins[particle_env]
    gx = torch.floor((local[:, 0] - grid_lower[0]) / cell_size).long()
    gy = torch.floor((local[:, 1] - grid_lower[1]) / cell_size).long()

    inside = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny) & torch.isfinite(local[:, 2])
    gx, gy = gx[inside], gy[inside]
    env = particle_env[inside]
    z = local[inside, 2]

    flat = torch.full((E * ny * nx,), floor_height, device=particle_pos_w.device, dtype=z.dtype)
    idx = (env * ny + gy) * nx + gx
    flat.scatter_reduce_(0, idx, z, reduce="amax", include_self=True)
    return flat.view(E, ny, nx)


def inside_drum(
    particle_pos_w: torch.Tensor,      # (P, 3)
    particle_env: torch.Tensor,        # (P,)
    drum_pos_w: torch.Tensor,          # (E, 3) body origin, on the spin axis
    drum_quat_w: torch.Tensor,         # (E, 4) x, y, z, w
    bore_radius: float,
    half_length: float,
    chord_z: float | None = None,
) -> torch.Tensor:
    """(P,) True for particles inside the drum of their env.

    Inside means, in the given body frame: within `bore_radius` of the spin
    axis (local y), within `half_length` along it, and, with `chord_z`, above
    local z = chord_z, the closed side of the chord across an inlet centred on
    local -z.
    """
    rel = particle_pos_w - drum_pos_w[particle_env]
    local = quat_apply_inverse(drum_quat_w[particle_env], rel)
    radial_sq = local[:, 0] ** 2 + local[:, 2] ** 2
    inside = (radial_sq < bore_radius ** 2) & (local[:, 1].abs() < half_length)
    if chord_z is not None:
        inside &= local[:, 2] > chord_z
    return inside


def drum_fill_mass(
    particle_pos_w: torch.Tensor,      # (P, 3)
    particle_env: torch.Tensor,        # (P,)
    particle_mass: float,              # kg per particle
    drum_pos_w: torch.Tensor,          # (E, 3) drum body origin, on the spin axis
    drum_quat_w: torch.Tensor,         # (E, 4) x, y, z, w
    bore_radius: float,
    half_length: float,
    num_envs: int,
    chord_z: float | None = None,
) -> torch.Tensor:
    """(E,) kg of particles inside each drum (inside_drum)."""
    inside = inside_drum(particle_pos_w, particle_env, drum_pos_w, drum_quat_w, bore_radius, half_length, chord_z)
    contrib = inside.to(particle_pos_w.dtype) * particle_mass

    out = torch.zeros(num_envs, device=particle_pos_w.device, dtype=contrib.dtype)
    out.scatter_add_(0, particle_env, contrib)
    return out


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v (N, 3) by the inverse of q (N, 4), quaternion order (x, y, z, w)."""
    xyz = q[:, :3]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return v - q[:, 3:4] * t + torch.cross(xyz, t, dim=-1)


def mpm_particle_state(mpm_object) -> tuple[torch.Tensor, torch.Tensor]:
    """(P, 3) world positions and (P,) env index of a Newton MPMObject's particles.

    MPMObject.data.particle_pos_w is (num_envs, particles_per_env) wp.vec3f.
    The positions are a zero-copy view of simulation memory, valid until the
    next step.
    """
    pos = mpm_object.data.particle_pos_w.torch          # (E, P, 3)
    num_envs, per_env = pos.shape[:2]
    env_idx = torch.arange(num_envs, device=pos.device).repeat_interleave(per_env)
    return pos.reshape(-1, 3), env_idx


def mpm_grid_resolution(lower, upper, voxel: float, per_cell: float) -> list[int]:
    """Particles along each axis of an MPMGridCfg lattice: ceil(per_cell * extent / voxel)."""
    return [max(math.ceil(per_cell * (hi - lo) / voxel), 1) for lo, hi in zip(lower, upper)]


def mpm_grid_particle_mass(cfg) -> float:
    """Per-particle mass [kg] of an MPMGridCfg, as isaaclab_newton's spawner
    computes it: the grid volume over the particle count, times density."""
    if cfg.mass is not None:
        return float(cfg.mass)

    extent = [hi - lo for lo, hi in zip(cfg.lower, cfg.upper)]
    resolution = mpm_grid_resolution(cfg.lower, cfg.upper, cfg.voxel_size, cfg.particles_per_cell)
    cell_volume = 1.0
    for e, r in zip(extent, resolution):
        cell_volume *= e / r
    return cell_volume * cfg.material.density
