# rft.py
#
# Analytical soil for the bucket drums, in place of MPM. Forces come from
# resistive force theory (RFT); the soil is a height grid.
#
# RFT (Li, Zhang & Goldman, Science 339, 1408, 2013): a flat patch pushed
# through grains feels a stress proportional to its depth below the surface.
# The stress per metre of depth depends on the patch's tilt and its direction
# of motion, times one scale for the soil.
#
# A drum is three sets of patches:
#   wall     the shroud's outer surface. Loaded where it faces its own motion.
#   opening  the gap in the wall. Soil passes through it into the drum, so it
#            is loaded only once the drum is full.
#   vanes    the rotor blades. Loaded where they show through the opening,
#            scaled by drum speed / tip speed: they meet new soil only as fast
#            as the drum moves into it.
# Depth is measured from the bed surface just ahead of the drum. Only motion
# in the vertical plane across the drum's axis is resisted; sliding along the
# axis is not.
#
# Soil inside the vane-tip circle comes off the grid into the drum until the
# drum is full. Its weight loads the shroud; carrying it round loads the rotor.
#
# Positions are env-local metres. Forces and torques are world-frame, torques
# about the drum's axis point. Tensors lead with (E, D): envs, drums.
#
#   once        patches = drum_patches()
#               cells = cell_centres(soil, nx, ny)
#   each step   loads = drum_loads(patches, <drum state>, height, fill, capacity, soil)
#               height, fill = excavate(height, fill, cells, centre, drum_axis(shroud_quat), capacity, soil)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..excavator import (
    POCKET_ARC,
    ROTOR_HALF_LEN,
    ROTOR_TIP_R,
    ROTOR_VANES,
    SHROUD_IN_R,
    SHROUD_INLET,
    SHROUD_OUT_R,
    vane_points,
)

V_STILL = 0.005     # m/s; below this a patch's force fades to zero
SPIN_STILL = 0.3    # rad/s; below this the carrying torque fades to zero


@dataclass
class Soil:
    density: float = 1500.0                     # kg/m3
    friction: float = 0.84                      # tan(friction angle)
    gravity: float = 1.62                       # m/s2
    cell: float = 0.04                          # height-grid cell, m
    lower: tuple[float, float] = (0.0, 0.0)     # env-local (x, y) of the grid's low corner

    @property
    def zeta(self) -> float:
        """RFT stress scale, N/m3. Set so a vertical plate pushed sideways feels
        passive earth pressure: tan^2(45 deg + phi/2) * density * gravity per
        metre of depth. 0.370 is alpha(pi/2, 0)[0]."""
        kp = math.tan(0.25 * math.pi + 0.5 * math.atan(self.friction)) ** 2
        return kp * self.density * self.gravity / 0.370


class Patches(NamedTuple):
    pos: torch.Tensor       # (P, 3) in the body frame
    normal: torch.Tensor    # (P, 3) unit
    area: torch.Tensor      # (P,) m2


def alpha(beta: torch.Tensor, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RFT stress per unit depth on a plate, (horizontal, vertical), as a
    fraction of the scale: the generic fit of Li et al., table S2.

    beta is the plate's tilt from level and gamma the direction it moves, both
    measured from the same horizontal direction: gamma = pi/2 is straight
    down, and beta > 0 raises the edge that leads at gamma = 0. The horizontal
    part acts against that direction, the vertical part upward. A level plate
    pushed straight down reads (0, 1).
    """
    vertical = (
        0.206 + 0.169 * torch.cos(2 * beta) + 0.212 * torch.sin(2 * beta + gamma)
        + 0.358 * torch.sin(gamma) + 0.055 * torch.sin(gamma - 2 * beta)
    )
    horizontal = (
        -0.124 * torch.cos(2 * beta + gamma) + 0.253 * torch.cos(gamma)
        + 0.007 * torch.cos(gamma - 2 * beta) + 0.088 * torch.sin(2 * beta)
    )
    return horizontal, vertical


def patch_force(normal, vel, area_depth, u, zeta: float) -> torch.Tensor:
    """RFT force (..., 3) on patches with unit `normal` moving at `vel`, given
    area * depth of each. u is the horizontal unit vector of the vertical
    plane they move in."""
    n_u, v_u = (normal * u).sum(-1), (vel * u).sum(-1)
    horizontal, vertical = alpha(torch.atan2(-n_u, normal[..., 2]), torch.atan2(-vel[..., 2], v_u))
    speed = torch.hypot(v_u, vel[..., 2])
    size = zeta * area_depth * speed / (speed + V_STILL)
    force = -(size * horizontal)[..., None] * u
    force[..., 2] += size * vertical
    return force


def rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors (..., 3) by quaternions (..., 4) in (x, y, z, w)."""
    xyz, vec = torch.broadcast_tensors(quat[..., :3], vec)
    t = 2.0 * torch.linalg.cross(xyz, vec)
    return vec + quat[..., 3:] * t + torch.linalg.cross(xyz, t)


def drum_axis(quat: torch.Tensor) -> torch.Tensor:
    """(..., 3) world direction of a drum's axis, its bodies' +y."""
    return rotate(quat, torch.tensor([0.0, 1.0, 0.0], device=quat.device))


def drum_patches(strips: int = 5, arcs: int = 36, device="cpu") -> tuple[Patches, torch.Tensor, Patches]:
    """(wall, opening, vanes) in the drum's body frames, axis along y.

    wall: patches round the shroud's whole outer circle, in the shroud frame.
    opening: (P,) True for the wall patches over the inlet, centred on -z.
    vanes: one patch per straight segment of each spiral vane, rotor frame.
    Each is repeated in `strips` strips along the axis.
    """
    y = ((torch.arange(strips) + 0.5) / strips * 2.0 - 1.0) * ROTOR_HALF_LEN
    width = 2.0 * ROTOR_HALF_LEN / strips

    def spread(xz, normal_xz, length) -> Patches:
        """(N, 2) in-plane patches of the given lengths, copied into every strip."""
        zero = torch.zeros_like(length)
        pos = torch.stack([xz[:, 0], zero, xz[:, 1]], -1)[:, None].repeat(1, strips, 1)
        pos[..., 1] = y
        normal = torch.stack([normal_xz[:, 0], zero, normal_xz[:, 1]], -1)[:, None].repeat(1, strips, 1)
        area = (length * width)[:, None].repeat(1, strips)
        return Patches(pos.reshape(-1, 3).to(device), normal.reshape(-1, 3).to(device), area.reshape(-1).to(device))

    phi = (torch.arange(arcs) + 0.5) / arcs * 2.0 * math.pi
    ring = torch.stack([phi.cos(), phi.sin()], -1)
    wall = spread(SHROUD_OUT_R * ring, ring, torch.full((arcs,), SHROUD_OUT_R * 2.0 * math.pi / arcs))
    opening = -wall.normal[:, 2] > math.cos(0.5 * SHROUD_INLET)

    points = torch.tensor([vane_points(i * POCKET_ARC) for i in range(ROTOR_VANES)])    # (vanes, points, 2)
    a, b = points[:, :-1].reshape(-1, 2), points[:, 1:].reshape(-1, 2)
    length = (b - a).norm(dim=-1)
    across = torch.stack([a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]], -1) / length[:, None]
    vanes = spread(0.5 * (a + b), across, length)
    return wall, opening, vanes


def cell_centres(soil: Soil, nx: int, ny: int, device="cpu") -> torch.Tensor:
    """(ny, nx, 2) env-local xy of every grid cell's centre."""
    x = soil.lower[0] + (torch.arange(nx, device=device) + 0.5) * soil.cell
    y = soil.lower[1] + (torch.arange(ny, device=device) + 0.5) * soil.cell
    return torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)


def height_at(height: torch.Tensor, soil: Soil, xy: torch.Tensor) -> torch.Tensor:
    """Soil height at env-local points xy (E, ..., 2) on the grid height
    (E, ny, nx). Off the grid there is no soil: 0."""
    E, ny, nx = height.shape
    ix = torch.floor((xy[..., 0] - soil.lower[0]) / soil.cell).long()
    iy = torch.floor((xy[..., 1] - soil.lower[1]) / soil.cell).long()
    on_grid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    index = (iy.clamp(0, ny - 1) * nx + ix.clamp(0, nx - 1)).reshape(E, -1)
    return height.reshape(E, -1).gather(1, index).reshape(ix.shape) * on_grid


def drum_loads(
    patches, centre, lin, shroud_quat, shroud_ang, rotor_quat, rotor_ang, height, fill, capacity, soil: Soil
):
    """Soil force and torque on each drum's shroud and rotor.

    patches: drum_patches(). centre, lin: (E, D, 3) the drum's axis point and
    its velocity. shroud_quat, rotor_quat (E, D, 4) and shroud_ang, rotor_ang
    (E, D, 3): orientation (x, y, z, w) and angular velocity of the two bodies.
    height (E, ny, nx): the soil grid. fill, capacity: (E, D) kg in the drum
    and the most it holds.

    Returns shroud force, shroud torque, rotor force, rotor torque: (E, D, 3) each.
    """
    wall, opening, vanes = patches
    up = torch.tensor([0.0, 0.0, 1.0], device=centre.device)
    axis = drum_axis(shroud_quat)
    # Horizontal, across the axis: with up, the plane the drum's patches move in.
    u = torch.nn.functional.normalize(torch.linalg.cross(up.expand_as(axis), axis), dim=-1)
    # From the axis point to the bed one cell beyond the drum's leading side.
    ahead = torch.sign((lin * u).sum(-1, keepdim=True)) * (SHROUD_OUT_R + soil.cell) * u

    def loads(p: Patches, quat, ang):
        """Offset from the axis point, normal, velocity and RFT force of every patch: (E, D, P, 3)."""
        arm = rotate(quat[..., None, :], p.pos)
        normal = rotate(quat[..., None, :], p.normal)
        vel = lin[..., None, :] + torch.linalg.cross(ang[..., None, :], arm)
        # Depth below the bed surface ahead of the drum, in the patch's own strip.
        strip = (arm * axis[..., None, :]).sum(-1, keepdim=True) * axis[..., None, :]
        surface = height_at(height, soil, (centre + ahead)[..., None, :2] + strip[..., :2])
        depth = (surface - centre[..., None, 2] - arm[..., 2]).clamp(min=0.0)
        return arm, normal, vel, patch_force(normal, vel, p.area * depth, u[..., None, :], soil.zeta)

    arm, normal, vel, force = loads(wall, shroud_quat, shroud_ang)
    faces_motion = (normal * vel).sum(-1) > 0.0
    full = (fill >= capacity)[..., None]
    force = force * (faces_motion & (~opening | full))[..., None]
    weight = fill * soil.gravity
    shroud_force = force.sum(-2)
    shroud_force[..., 2] -= weight
    shroud_torque = torch.linalg.cross(arm, force).sum(-2)

    arm, normal, vel, force = loads(vanes, rotor_quat, rotor_ang)
    inlet = rotate(shroud_quat, -up)
    radial = arm - (arm * axis[..., None, :]).sum(-1, keepdim=True) * axis[..., None, :]
    radial = torch.nn.functional.normalize(radial, dim=-1)
    shows = (radial * inlet[..., None, :]).sum(-1) > math.cos(0.5 * SHROUD_INLET)
    spin = (rotor_ang * axis).sum(-1)
    fresh = (lin.norm(dim=-1) / (spin.abs() * ROTOR_TIP_R + 1e-6)).clamp(max=1.0)
    force = force * (shows * fresh[..., None])[..., None]
    # The load lies as a segment of the bore tilted to its angle of repose; its
    # centroid is about 0.87 R (1 - fill / capacity) off the axis.
    carry = weight * 0.87 * SHROUD_IN_R * (1.0 - fill / capacity) * math.sin(math.atan(soil.friction))
    carry = carry * spin / (spin.abs() + SPIN_STILL)
    rotor_torque = torch.linalg.cross(arm, force).sum(-2) - carry[..., None] * axis
    return shroud_force, shroud_torque, force.sum(-2), rotor_torque


def excavate(height, fill, cells, centre, axis, capacity, soil: Soil) -> tuple[torch.Tensor, torch.Tensor]:
    """Move the soil inside each drum's vane-tip circle off the grid into the
    drum, as far as the drum has room.

    height (E, ny, nx), fill (E, D), cells from cell_centres(), centre and
    axis (E, D, 3) the drum's axis point and drum_axis(). Returns the new
    height and fill.
    """
    rel = cells - centre[..., None, None, :2]                           # (E, D, ny, nx, 2)
    along = (rel * axis[..., None, None, :2]).sum(-1)
    across = rel[..., 0] * axis[..., None, None, 1] - rel[..., 1] * axis[..., None, None, 0]
    under = (along.abs() <= ROTOR_HALF_LEN) & (across.abs() < ROTOR_TIP_R)
    lowest = centre[..., None, None, 2] - (ROTOR_TIP_R ** 2 - across ** 2).clamp(min=0.0).sqrt()
    excess = (height[:, None] - lowest).clamp(min=0.0) * under          # m of soil above the tip circle
    offered = excess.sum((-2, -1)) * soil.cell ** 2 * soil.density      # (E, D) kg
    taken = torch.minimum(offered, capacity - fill)
    share = taken / offered.clamp(min=1e-9)
    return height - (share[..., None, None] * excess).sum(1), fill + taken
