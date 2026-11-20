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
# RFT is for dry, cohesionless grains. Cohesion c is added the classical way: a
# cohesive soil behaves like a cohesionless one under an extra pressure, which
# here is an extra depth 2c / (sqrt(Kp) rho g) for every patch in the soil. It
# turns Rankine's passive pressure Kp rho g z into Kp rho g z + 2c sqrt(Kp).
#
# A drum is three sets of patches:
#   wall     the shroud's outer surface. Loaded where it faces its own motion.
#   opening  the gap in the wall. Soil passes through it into the drum, so it
#            is loaded only by the soil the drum cannot take: all once it is
#            full, in part while it takes less than it reaches.
#   vanes    the rotor blades. Loaded where they show through the opening,
#            scaled by drum speed / tip speed: they meet new soil only as fast
#            as the drum moves into it.
# Depth is measured from the bed surface just ahead of the drum. Only motion
# in the vertical plane across the drum's axis is resisted; sliding along the
# axis is not.
#
# Soil inside the vane-tip circle comes off the grid into the drum, at most
# soil.intake kg/s and until the drum is full. Its weight loads the shroud;
# carrying it round loads the rotor. With soil.repose the soil it does not take
# is shoved out of the circle, ahead of the shroud or behind it, as loose soil:
# loose soil runs downhill wherever it stands steeper than its angle of repose,
# but not into a drum standing in the soil, while the undisturbed bed, cohesive,
# keeps its shape.
#
# A wheel is the tyre and its grousers, loaded the same way where they face
# their own motion. Its depth is measured from the higher of the bed surfaces
# just ahead of and just behind it, so a wheel stands on the soil beside its
# own rut. The soil under a tyre is pressed down to it: compacted, not moved.
#
# Positions are env-local metres. Forces and torques are world-frame, torques
# about the drum's or wheel's axis point. Tensors lead with (E, D): envs, drums
# (or wheels).
#
#   once        patches = drum_patches()
#               cells = cell_centres(soil, nx, ny)
#   each step   loads = drum_loads(patches, <drum state>, height, fill, capacity, left, soil)
#               height, loose, fill, left = excavate(height, loose, fill, cells, centre,
#                                                    drum_axis(shroud_quat), capacity, dt, soil)
#               height, loose = slump(height, loose, drum_body(cells, centre, axis, height), soil)
#   wheels      wheels = wheel_patches()
#               force, torque = wheel_loads(wheels, <wheel state>, height, soil)
#               height = press(height, cells, centre, drum_axis(wheel_quat))

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..excavator import (
    GROUSER_HALF_T,
    GROUSER_PROTRUDE,
    POCKET_ARC,
    ROTOR_HALF_LEN,
    ROTOR_TIP_R,
    ROTOR_VANES,
    SHROUD_IN_R,
    SHROUD_INLET,
    SHROUD_OUT_R,
    WHEEL_GROUSERS,
    WHEEL_HALF_W,
    WHEEL_RADIUS,
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
    cohesion: float = 0.0                       # Pa, Mohr-Coulomb c
    scale: float = 6.15                         # stress per Kp rho g; see zeta
    intake: float = math.inf                    # kg/s a drum takes at most; inf: all it reaches
    repose: float = 0.0                         # tan(angle of repose) of loose soil; 0: none is shoved

    @property
    def kp(self) -> float:
        """Passive earth pressure coefficient, tan^2(45 deg + phi/2)."""
        return math.tan(0.25 * math.pi + 0.5 * math.atan(self.friction)) ** 2

    @property
    def zeta(self) -> float:
        """RFT stress scale, N/m3: scale * Kp * density * gravity. The default
        6.15 is what the three glass-sphere beds of Li et al. measure (5.5, 6.3,
        6.6), from their scales, bulk densities and angles of repose."""
        return self.scale * self.kp * self.density * self.gravity

    @property
    def cohesion_depth(self) -> float:
        """Extra depth that stands in for cohesion, m: 2c / (sqrt(Kp) * density * gravity)."""
        return 2.0 * self.cohesion / (math.sqrt(self.kp) * self.density * self.gravity)


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
    """(..., 3) world direction of a drum's or wheel's axis, its bodies' +y."""
    return rotate(quat, torch.tensor([0.0, 1.0, 0.0], device=quat.device))


def strips_of(xz, normal_xz, length, strips: int, half_len: float, device) -> Patches:
    """(N, 2) patches in the x-z plane, of the given lengths, copied into
    `strips` strips across 2 half_len along y."""
    y = ((torch.arange(strips) + 0.5) / strips * 2.0 - 1.0) * half_len
    width = 2.0 * half_len / strips
    zero = torch.zeros_like(length)
    pos = torch.stack([xz[:, 0], zero, xz[:, 1]], -1)[:, None].repeat(1, strips, 1)
    pos[..., 1] = y
    normal = torch.stack([normal_xz[:, 0], zero, normal_xz[:, 1]], -1)[:, None].repeat(1, strips, 1)
    area = (length * width)[:, None].repeat(1, strips)
    return Patches(pos.reshape(-1, 3).to(device), normal.reshape(-1, 3).to(device), area.reshape(-1).to(device))


def drum_patches(
    strips: int = 5, arcs: int = 72, pieces: int = 4, device="cpu"
) -> tuple[Patches, torch.Tensor, Patches]:
    """(wall, opening, vanes) in the drum's body frames, axis along y.

    wall: `arcs` patches round the shroud's whole outer circle, shroud frame.
    opening: (P,) True for the wall patches over the inlet, centred on -z.
    vanes: each straight segment of each spiral vane in `pieces` patches,
    rotor frame. Each is repeated in `strips` strips along the axis.
    """

    def spread(xz, normal_xz, length) -> Patches:
        return strips_of(xz, normal_xz, length, strips, ROTOR_HALF_LEN, device)

    phi = (torch.arange(arcs) + 0.5) / arcs * 2.0 * math.pi
    ring = torch.stack([phi.cos(), phi.sin()], -1)
    wall = spread(SHROUD_OUT_R * ring, ring, torch.full((arcs,), SHROUD_OUT_R * 2.0 * math.pi / arcs))
    opening = -wall.normal[:, 2] > math.cos(0.5 * SHROUD_INLET)

    points = torch.tensor([vane_points(i * POCKET_ARC) for i in range(ROTOR_VANES)])    # (vanes, points, 2)
    a, b = points[:, :-1].reshape(-1, 2), points[:, 1:].reshape(-1, 2)
    length = (b - a).norm(dim=-1)
    across = torch.stack([a[:, 1] - b[:, 1], b[:, 0] - a[:, 0]], -1) / length[:, None]
    step = (torch.arange(pieces) + 0.5) / pieces
    middle = (a[:, None] + step[None, :, None] * (b - a)[:, None]).reshape(-1, 2)
    vanes = spread(middle, across.repeat_interleave(pieces, 0), (length / pieces).repeat_interleave(pieces))
    return wall, opening, vanes


def wheel_patches(strips: int = 4, arcs: int = 72, device="cpu") -> Patches:
    """A wheel in its body frame, axis along y: `arcs` patches round the tyre,
    and each grouser's two faces and tip, in `strips` strips across the tyre."""
    phi = (torch.arange(arcs) + 0.5) / arcs * 2.0 * math.pi
    ring = torch.stack([phi.cos(), phi.sin()], -1)
    g = torch.arange(WHEEL_GROUSERS) * 2.0 * math.pi / WHEEL_GROUSERS
    radial = torch.stack([g.cos(), g.sin()], -1)
    across = torch.stack([-g.sin(), g.cos()], -1)
    mid = (WHEEL_RADIUS + 0.5 * GROUSER_PROTRUDE) * radial
    xz = torch.cat([
        WHEEL_RADIUS * ring,
        mid + GROUSER_HALF_T * across,
        mid - GROUSER_HALF_T * across,
        (WHEEL_RADIUS + GROUSER_PROTRUDE) * radial,
    ])
    normal = torch.cat([ring, across, -across, radial])
    length = torch.cat([
        torch.full((arcs,), WHEEL_RADIUS * 2.0 * math.pi / arcs),
        torch.full((2 * WHEEL_GROUSERS,), GROUSER_PROTRUDE),
        torch.full((WHEEL_GROUSERS,), 2.0 * GROUSER_HALF_T),
    ])
    return strips_of(xz, normal, length, strips, WHEEL_HALF_W, device)


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
    patches, centre, lin, shroud_quat, shroud_ang, rotor_quat, rotor_ang, height, fill, capacity, left, soil: Soil
):
    """Soil force and torque on each drum's shroud and rotor.

    patches: drum_patches(). centre, lin: (E, D, 3) the drum's axis point and
    its velocity. shroud_quat, rotor_quat (E, D, 4) and shroud_ang, rotor_ang
    (E, D, 3): orientation (x, y, z, w) and angular velocity of the two bodies.
    height (E, ny, nx): the soil grid. fill, capacity: (E, D) kg in the drum
    and the most it holds. left (E, D): the share of the soil it reached that
    excavate() last left, which loads the opening.

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
        depth = torch.where(depth > 0.0, depth + soil.cohesion_depth, depth)
        return arm, normal, vel, patch_force(normal, vel, p.area * depth, u[..., None, :], soil.zeta)

    arm, normal, vel, force = loads(wall, shroud_quat, shroud_ang)
    faces_motion = (normal * vel).sum(-1) > 0.0
    blocked = torch.maximum((fill >= capacity).float(), left)[..., None]
    force = force * (faces_motion * torch.where(opening, blocked, 1.0))[..., None]
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
    # Carried round by the vanes, the load sits tilted to its angle of repose.
    # Its centroid is on the chord across the opening when the drum is near
    # empty and on the axis when it is full.
    lever = math.cos(0.5 * SHROUD_INLET) * SHROUD_IN_R * (1.0 - fill / capacity)
    carry = weight * lever * math.sin(math.atan(soil.friction))
    carry = carry * spin / (spin.abs() + SPIN_STILL)
    rotor_torque = torch.linalg.cross(arm, force).sum(-2) - carry[..., None] * axis
    return shroud_force, shroud_torque, force.sum(-2), rotor_torque


def excavate(height, loose, fill, cells, centre, axis, capacity, dt: float, soil: Soil):
    """Move the soil inside each drum's vane-tip circle off the grid into the
    drum, at most soil.intake * dt and as far as it has room. With soil.repose
    the soil it leaves is shoved out of the circle as loose soil; without, it
    stays where it is.

    height, loose (E, ny, nx): the grid and the loose soil on it. fill (E, D),
    cells from cell_centres(), centre and axis (E, D, 3) the drum's axis point and
    drum_axis(). Returns height, loose, fill, and the share of the soil each drum
    reached that it left (E, D).
    """
    under, lowest, along, across = beneath(cells, centre, axis, ROTOR_TIP_R, ROTOR_HALF_LEN)
    excess = (height[:, None] - lowest).clamp(min=0.0) * under          # m of soil above the tip circle
    offered = excess.sum((-2, -1)) * soil.cell ** 2 * soil.density      # (E, D) kg
    taken = torch.minimum(torch.minimum(offered, capacity - fill), torch.full_like(offered, soil.intake * dt))
    share = taken / offered.clamp(min=1e-9)
    gone = share[..., None, None] * excess
    if soil.repose:
        shoved = shove(excess - gone, along, across, centre, axis, soil)
        gone = excess
    gone = gone.sum(1)
    height, loose = height - gone, (loose - gone).clamp(min=0.0)         # loose soil lies on top
    if soil.repose:
        height, loose = height + shoved, loose + shoved
    return height, loose, fill + taken, (1.0 - share) * (offered > 0.0)


def beneath(cells, centre, axis, radius: float, half_len: float):
    """Which grid cells lie under each cylinder of `radius` and `half_len`, the
    cylinder's lowest point over each, and each cell's offset along the axis
    and across it: (E, D, ny, nx) each."""
    rel = cells - centre[..., None, None, :2]                           # (E, D, ny, nx, 2)
    along = (rel * axis[..., None, None, :2]).sum(-1)
    across = rel[..., 0] * axis[..., None, None, 1] - rel[..., 1] * axis[..., None, None, 0]
    under = (along.abs() <= half_len) & (across.abs() < radius)
    lowest = centre[..., None, None, 2] - (radius ** 2 - across ** 2).clamp(min=0.0).sqrt()
    return under, lowest, along, across


def shove(soil_left, along, across, centre, axis, soil: Soil) -> torch.Tensor:
    """The height (E, ny, nx) that soil_left (E, D, ny, nx), in each drum's
    vane-tip circle, adds where it is shoved: the cell just outside the shroud
    on the side of the axis it lay, in its own lane along the axis."""
    E, D, ny, nx = soil_left.shape
    side = torch.stack([axis[..., 1], -axis[..., 0]], -1)[..., None, None, :]   # unit +across
    reach = torch.sign(across)[..., None] * (SHROUD_OUT_R + 0.5 * soil.cell)
    xy = centre[..., None, None, :2] + along[..., None] * axis[..., None, None, :2] + reach * side
    ix = torch.floor((xy[..., 0] - soil.lower[0]) / soil.cell).long().clamp(0, nx - 1)
    iy = torch.floor((xy[..., 1] - soil.lower[1]) / soil.cell).long().clamp(0, ny - 1)
    out = torch.zeros(E, ny * nx, device=soil_left.device)
    return out.scatter_add_(1, (iy * nx + ix).reshape(E, -1), soil_left.reshape(E, -1)).reshape(E, ny, nx)


# Each pair of neighbouring cells once: (row, column) offsets.
NEIGHBOURS = ((0, 1), (1, 0), (1, 1), (1, -1))


def drum_body(cells, centre, axis, height) -> torch.Tensor:
    """The grid cells (E, ny, nx) where a drum stands in the soil: under its
    shroud, which reaches below their surface."""
    under, lowest, _, _ = beneath(cells, centre, axis, SHROUD_OUT_R, ROTOR_HALF_LEN)
    return (under & (lowest < height[:, None])).any(1)


def slump(height, loose, body, soil: Soil, sweeps: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Let the loose soil (E, ny, nx), part of height, run downhill wherever two
    neighbouring cells differ by more than soil.repose allows, but not into or
    out of a body cell (E, ny, nx), where a drum stands: it leans on the drum.
    Each pair moves a quarter of its excess per sweep, and a cell gives each of
    its two neighbours in a direction at most half its loose soil."""
    height, loose = height.clone(), loose.clone()
    ny, nx = height.shape[-2:]
    for _ in range(sweeps):
        for dy, dx in NEIGHBOURS:
            here = (slice(None), slice(0, ny - dy), slice(max(0, -dx), nx - max(0, dx)))
            there = (slice(None), slice(dy, ny), slice(max(0, dx), nx - max(0, -dx)))
            step = height[here] - height[there]
            over = 0.25 * (step.abs() - soil.repose * soil.cell * math.hypot(dy, dx)).clamp(min=0.0)
            flow = torch.where(step > 0.0, torch.minimum(over, 0.5 * loose[here]), -torch.minimum(over, 0.5 * loose[there]))
            flow = torch.where(body[here] | body[there], 0.0, flow)
            height[here] -= flow
            height[there] += flow
            loose[here] -= flow
            loose[there] += flow
    return height, loose


def wheel_loads(patches: Patches, centre, lin, quat, ang, height, soil: Soil) -> tuple[torch.Tensor, torch.Tensor]:
    """Soil force and torque on each wheel, (E, W, 3) each.

    patches: wheel_patches(). centre, lin, ang: (E, W, 3) the wheel's axis
    point, velocity and angular velocity; quat (E, W, 4) its orientation.
    height (E, ny, nx): the soil grid.
    """
    up = torch.tensor([0.0, 0.0, 1.0], device=centre.device)
    axis = drum_axis(quat)
    u = torch.nn.functional.normalize(torch.linalg.cross(up.expand_as(axis), axis), dim=-1)
    arm = rotate(quat[..., None, :], patches.pos)
    normal = rotate(quat[..., None, :], patches.normal)
    vel = lin[..., None, :] + torch.linalg.cross(ang[..., None, :], arm)
    # Depth below the higher of the surfaces one cell beyond the grousers, ahead
    # and behind, in the patch's own strip.
    strip = (arm * axis[..., None, :]).sum(-1, keepdim=True) * axis[..., None, :]
    xy = centre[..., None, :2] + strip[..., :2]
    reach = ((WHEEL_RADIUS + GROUSER_PROTRUDE + soil.cell) * u)[..., None, :2]
    surface = torch.maximum(height_at(height, soil, xy + reach), height_at(height, soil, xy - reach))
    depth = (surface - centre[..., None, 2] - arm[..., 2]).clamp(min=0.0)
    depth = torch.where(depth > 0.0, depth + soil.cohesion_depth, depth)
    force = patch_force(normal, vel, patches.area * depth, u[..., None, :], soil.zeta)
    force = force * ((normal * vel).sum(-1) > 0.0)[..., None]
    return force.sum(-2), torch.linalg.cross(arm, force).sum(-2)


def press(height, cells, centre, axis) -> torch.Tensor:
    """The grid (E, ny, nx) with the soil under each wheel's tyre pressed down
    to it. centre and axis (E, W, 3): the wheels' axis points and drum_axis()."""
    under, lowest, _, _ = beneath(cells, centre, axis, WHEEL_RADIUS, WHEEL_HALF_W)
    tyre = torch.where(under, lowest, torch.full_like(lowest, math.inf)).amin(1)
    return torch.minimum(height, tyre)
