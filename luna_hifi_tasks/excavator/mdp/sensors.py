# sensors.py
#
# The two quantities that every excavator policy is built on:
#
#   1. soil_heightmap   -- MPM particles rasterised into a per-env height grid
#   2. drum_fill        -- mass of MPM particles inside a drum's cavity
#
# Both are pure PyTorch, deliberately. The obvious alternative is a Warp
# kernel, and the cost is the same either way (one pass over the particles,
# which is nothing next to the MPM solve itself). But Warp JIT-compiles to a
# cache directory, and on a cluster that cache is either cold on every job or
# it is a shared-filesystem coordination problem. scatter_reduce has neither
# failure mode. Revisit only if a profile says this is hot.
#
# Nothing here imports isaaclab or newton. That is on purpose: it means the
# whole file is testable on a laptop with no GPU, and it keeps the Newton API
# surface confined to two small adapters at the bottom, both of which are now
# checked against the isaaclab_newton source rather than guessed.

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
    particle read `floor_height` -- which must be the height of the hard floor
    under the bed, not zero, or an excavated cell and an empty cell become
    indistinguishable to the policy.

    Particles outside the grid are dropped rather than clamped: clamping piles
    everything outside the window onto the boundary cells and puts a fake wall
    of soil at the edge of the observation.
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

    Heights are made RELATIVE to a reference (the chassis height, or the
    undisturbed bed surface) before scaling. An absolute height observation
    makes the policy relearn the task at every terrain elevation.
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

    A particle counts when, in the DRUM's own frame, it is within `bore_radius`
    of the spin axis and within `half_length` along it. The drum's frame is the
    right one to test in: the drum both pitches with the arm and spins, so a
    world-axis-aligned box test would drift off the cavity as soon as the arm
    moved.

    The spin axis is the drum body's local y (see excavator.py -- every hinge in
    this machine turns about y), so the radial test uses the local x and z
    components.
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
    """Normalise fill to [0, 1+]. Not clamped at the top on purpose: a reading
    above 1.0 means the bore capacity estimate is wrong, or particles are
    inside the shell wall, and silently clamping hides both."""
    return fill_mass / capacity_kg


# ---------------------------------------------------------------------------
# Quaternion helper
#
# ORDER IS (x, y, z, w). Isaac Lab 3.x migrated from 2.x's (w, x, y, z), and
# every pose in isaaclab.assets on this branch is xyzw -- base_articulation_data
# documents it ten times and wxyz zero times. Getting this backwards does not
# crash: it silently rotates by a different orientation, so drum fill reads
# plausible-but-wrong numbers and the reward quietly trains the wrong thing.
#
# Mirrors isaaclab.utils.math.quat_apply_inverse exactly, duplicated so this
# module stays importable without isaaclab.
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

    Verified against isaaclab_newton on the develop branch:

        MPMObject.data.particle_pos_w   ProxyArray, wp.vec3f,
                                        shape (num_instances, particles_per_object)
        MPMObject.num_instances         int
        MPMObject.particles_per_object  int

    Two details worth knowing. Particles are ALREADY shaped per-environment, so
    the env index is a repeat_interleave over a fixed stride and not the
    "divide a flat array" guess it would be natural to write. And `.torch` is a
    zero-copy view onto the warp array, so this costs a reshape, not a device
    round-trip -- but it also means the tensor aliases live simulation memory
    and must not be held across a step.

    There is deliberately no mass here: MPMObjectData exposes no per-particle
    mass at all. Use `mpm_grid_particle_mass` on the spawn cfg instead.
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
    """Per-particle mass [kg] for an MPMGridCfg, by the same arithmetic the
    spawner uses.

    Mirrors isaaclab_newton.sim.spawners.mpm.mpm exactly, because drum fill is
    reported in kilograms and a mass that disagrees with the spawner's makes
    every excavation reward wrong by a constant factor -- which trains a
    perfectly confident policy toward a miscalibrated target.

    Note the lattice resolution is ceil()ed per axis, so cell_volume is NOT
    simply voxel_size**3 / particles_per_cell. Rounding up on a bed whose
    extent is not a whole number of voxels makes the real particles smaller
    than the naive formula suggests.
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
