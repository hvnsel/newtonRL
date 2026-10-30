# mdp/paths.py
#
# Distance to the goal the way the machine has to drive it, round the craters
# and boulders rather than through them, for the navigate reward.
#
# Each cell's ground is read on a square grid. A grid point is an obstacle
# where the ground steps to a neighbour by more than a threshold: a boulder's
# side, a crater's wall. Every grid point gets a travel cost per metre: a lot
# on an obstacle, rising towards it across a zone round it, 1 on open ground
# beyond. Costs are finite, so a machine that has come into a zone, or
# worse, still reads a distance with a slope that leads out.
#
# path_field spreads the cheapest cost to the goal over the whole grid from
# the goal outwards: Bellman-Ford over a 16-neighbour stencil, every grid
# point at once, batched over envs. The stencil's knight moves keep the
# distance on open ground within 3% of the straight line in any direction,
# where 8 neighbours run up to 8% long at 22.5 degrees. A move is charged at
# the cost of the point it leaves, and no move spans more than two points
# along an axis; an obstacle's ring is at least two points thick, since both
# sides of a step are marked, so no move hops it.
#
# Only the reward reads these maps. The policy never sees them.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# (di, dj) of the 16 neighbours, and the centre, which keeps a point's own
# value. Kernel index into a 5 x 5 unfold is (di + 2) * 5 + (dj + 2).
_STENCIL = [(0, 0)] + [
    (di, dj)
    for di in range(-2, 3)
    for dj in range(-2, 3)
    if (abs(di), abs(dj)) in ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2))
]
_STENCIL_IDX = [(di + 2) * 5 + (dj + 2) for di, dj in _STENCIL]
_STENCIL_LEN = [math.hypot(di, dj) for di, dj in _STENCIL]

# Larger than any path cost a grid can hold, and finite, so nothing in the
# relaxation turns into inf - inf.
_FAR = 1.0e9


def grid_axis(size: float, res: float, device="cpu") -> torch.Tensor:
    """Grid point coordinates along one axis of a cell `size` metres wide,
    centred on the cell: -size/2 + res/2, ..., size/2 - res/2."""
    n = int(round(size / res))
    return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * res - 0.5 * size


def obstacle_cost(
    heights: torch.Tensor,       # (B, G, G) ground height on the grid, NaN where unknown
    res: float,
    step: float,
    zone: float,
    zone_cost: float,
    obstacle_cost: float,
) -> torch.Tensor:
    """(B, G, G) travel cost per metre: obstacle_cost on both sides of every
    step taller than `step` between axis neighbours; across `zone` metres
    round one, falling linearly from zone_cost beside it to 1 at the zone's
    edge; 1 beyond."""
    h = heights
    dx = torch.nan_to_num((h[:, 1:, :] - h[:, :-1, :]).abs(), nan=0.0) > step
    dy = torch.nan_to_num((h[:, :, 1:] - h[:, :, :-1]).abs(), nan=0.0) > step
    obst = torch.zeros_like(h, dtype=torch.bool)
    obst[:, 1:, :] |= dx
    obst[:, :-1, :] |= dx
    obst[:, :, 1:] |= dy
    obst[:, :, :-1] |= dy

    # Distance to the nearest obstacle point, out to the zone's edge: the
    # nearest of every offset within it at which one sits.
    k = int(math.ceil(zone / res))
    pad = F.pad(obst.float(), (k, k, k, k)).bool()
    gx, gy = h.shape[1], h.shape[2]
    near = torch.full_like(h, float("inf"))
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            r = math.hypot(di, dj) * res
            if r > zone:
                continue
            hit = pad[:, k + di:k + di + gx, k + dj:k + dj + gy]
            near = torch.where(hit, torch.minimum(near, torch.full_like(near, r)), near)

    rise = (1.0 - near / zone).clamp(0.0, 1.0)
    cost = 1.0 + (zone_cost - 1.0) * rise
    return torch.where(obst, torch.full_like(cost, obstacle_cost), cost)


def path_field(
    cost: torch.Tensor,          # (n, G, G) travel cost per metre
    goal_uv: torch.Tensor,       # (n, 2) goal in grid units: (x - axis[0]) / res
    res: float,
    max_iters: int = 256,
    check_every: int = 16,
    chunk: int = 128,
) -> torch.Tensor:
    """(n, G, G) cheapest cost from each grid point to the goal, in metres of
    open ground. Grid points within one spacing of the goal start at their
    straight distance to it; everything else relaxes from there until no
    value changes over check_every sweeps, or max_iters."""
    out = []
    for s in range(0, cost.shape[0], chunk):
        out.append(_path_field(cost[s:s + chunk], goal_uv[s:s + chunk], res, max_iters, check_every))
    return torch.cat(out, dim=0)


def _path_field(cost, goal_uv, res, max_iters, check_every):
    n, gx, gy = cost.shape
    dev = cost.device
    i = torch.arange(gx, device=dev, dtype=torch.float32).view(1, gx, 1)
    j = torch.arange(gy, device=dev, dtype=torch.float32).view(1, 1, gy)
    seed = torch.sqrt((i - goal_uv[:, 0].view(n, 1, 1)) ** 2 + (j - goal_uv[:, 1].view(n, 1, 1)) ** 2)
    d = torch.where(seed <= 1.0 + 1e-6, seed * res * cost, torch.full_like(cost, _FAR))

    idx = torch.tensor(_STENCIL_IDX, device=dev)
    step = torch.tensor(_STENCIL_LEN, device=dev).view(1, -1, 1) * res * cost.reshape(n, 1, gx * gy)
    flat = d.reshape(n, gx * gy)
    for it in range(max_iters):
        nb = F.unfold(F.pad(flat.view(n, 1, gx, gy), (2, 2, 2, 2), value=_FAR), 5)[:, idx]
        new = (nb + step).amin(dim=1)
        settled = (it + 1) % check_every == 0 and not bool((new < flat - 1e-5).any())
        flat = new
        if settled:
            break
    return flat.view(n, gx, gy)


def sample_field(field: torch.Tensor, uv: torch.Tensor, res: float) -> torch.Tensor:
    """(n,) the field at (n, 2) grid coordinates, bilinear. Off the grid the
    nearest edge value plus the straight distance out to the point."""
    n, gx, gy = field.shape
    u = uv[:, 0].clamp(0.0, gx - 1.0)
    v = uv[:, 1].clamp(0.0, gy - 1.0)
    out = torch.hypot(uv[:, 0] - u, uv[:, 1] - v) * res
    i0 = u.floor().long().clamp(max=gx - 2)
    j0 = v.floor().long().clamp(max=gy - 2)
    fu, fv = u - i0, v - j0
    rows = torch.arange(n, device=field.device)
    f00 = field[rows, i0, j0]
    f10 = field[rows, i0 + 1, j0]
    f01 = field[rows, i0, j0 + 1]
    f11 = field[rows, i0 + 1, j0 + 1]
    return (1 - fu) * (1 - fv) * f00 + fu * (1 - fv) * f10 + (1 - fu) * fv * f01 + fu * fv * f11 + out
