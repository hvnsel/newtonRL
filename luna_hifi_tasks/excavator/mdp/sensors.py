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
# surface confined to one small adapter (see particle_state_adapter below).

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
    drum_quat_w: torch.Tensor,         # (E, 4) w,x,y,z
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
    local = quat_rotate_inverse(drum_quat_w[particle_env], rel)

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
# Quaternion helper (w, x, y, z -- Isaac Lab's convention)
# ---------------------------------------------------------------------------


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v by the inverse of q. Same formula isaaclab.utils.math uses;
    duplicated here so this module stays importable without isaaclab."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    a = v * (2.0 * w * w - 1.0)
    b = torch.cross(xyz, v, dim=-1) * w * 2.0
    c = xyz * torch.sum(xyz * v, dim=-1, keepdim=True) * 2.0
    return a - b + c


# ---------------------------------------------------------------------------
# Newton adapter
# ---------------------------------------------------------------------------


def particle_state_adapter(mpm_object) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Pull (positions, env index, per-particle mass) out of a Newton MPMObject.

    THIS IS THE ONE FUNCTION WHOSE API IS UNVERIFIED. Everything else in this
    file is plain tensor maths that is tested in tests/test_sensors.py; this
    reaches into isaaclab_newton and the exact attribute names on
    MPMObject.data were not checkable offline.

    Check, in order:
      * the positions attribute -- likely `mpm_object.data.particle_pos_w` or
        `.particle_q`; it must be world-frame (P, 3)
      * whether particles are stored flat across envs (then env index is
        arange(P) // particles_per_env) or already shaped (E, P_per_env, 3),
        in which case flatten and build the index with repeat_interleave
      * per-particle mass: if Newton exposes only material density, mass is
        density * voxel_size**3 / particles_per_cell

    Get this wrong and drum_fill silently reads zero, which looks exactly like
    a policy that has not learned to dig yet. Assert on it once at startup:
    spawn the bed, drop the drum into it, and check fill goes up.
    """
    data = mpm_object.data
    pos = getattr(data, "particle_pos_w", None)
    if pos is None:
        pos = data.particle_q
    pos = pos.view(-1, 3)

    num_envs = mpm_object.num_instances
    per_env = pos.shape[0] // num_envs
    env_idx = torch.arange(num_envs, device=pos.device).repeat_interleave(per_env)

    mass = getattr(data, "particle_mass", None)
    if mass is None:
        raise RuntimeError(
            "MPMObject.data has no particle_mass; compute it from the material "
            "density and voxel size in the env cfg and pass it explicitly."
        )
    return pos, env_idx, mass
