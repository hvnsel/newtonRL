"""Heightfield site model for the excavation planner.

The site is a grid of patches, each the size of the dig policy's work area and
carrying an integer height in voxels. One trim is one dig pass: it clears one
voxel of depth from one patch (`height -= 1`), then the spoil goes to the dump.
Pure Python; no simulation. site_env.py runs the trims on the RFT soil and
reads each patch's height back from the bed.

The baselines choose the order of trims. The machine drives between
neighbouring patches whose height difference is within `climb_voxels`, and
digs a patch it straddles from the ground beside it down to `reach_voxels`.

Run it for a demo on a pit:

    python luna_hifi_tasks/planner/site.py
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field

Patch = tuple[int, int]      # (iy, ix)
Grid = list[list[float]]     # heights in voxels; a measured trim leaves fractions


@dataclass
class SiteCfg:
    # --- geometry -----------------------------------------------------
    voxel: float = 0.05
    """Metres per voxel; the depth of one trim."""

    patch_x: float = 0.80
    patch_y: float = 0.95
    """Trim footprint, metres: the dig's work area, 0.60 m ahead of the anchor
    plus 0.20 behind, by the 0.95 m drum swath."""

    # --- what a trim does ---------------------------------------------
    soil_density: float = 1500.0
    """kg/m3."""

    capture_fraction: float = 0.80
    """Share of the soil a trim cuts that ends up in the drum (the dig
    policy's Episode/payload_kg over Episode/removed_kg)."""

    drum_capacity: float = 150.0
    """kg the front drum holds."""

    trim_seconds: float = 20.0
    """Cutting time for one trim, seconds."""

    trim_success: float = 0.66
    """Probability a trim meets its shape tolerance (the dig policy's
    Episode/shape_done)."""

    # --- travel -------------------------------------------------------
    drive_speed: float = 0.5
    """m/s while hauling. Estimate."""

    dump_seconds: float = 6.0
    """Time to discharge, seconds."""

    # --- reachability -------------------------------------------------
    climb_degrees: float = 5.7
    """Steepest slope the laden machine drives, degrees: the dig's 0.10."""

    reach: float = 0.18
    """How far below its wheels the drum digs, metres: at the arm's limit
    the vane tips are 0.186 m below the wheel plane."""

    # --- trims on the soil, read back ---------------------------------
    done_tol: float = 0.015
    """A patch within this of its target is done, metres: the dig leaves its
    floor a centimetre or two off the plane."""

    progress: float = 0.01
    """A trim that lowers its patch less than this failed, metres."""

    retries: int = 3
    """A patch that fails this many trims in a row is set aside."""

    def __post_init__(self) -> None:
        self.patch_area = self.patch_x * self.patch_y
        self.trim_mass = self.patch_area * self.voxel * self.soil_density
        self.trim_capture = self.trim_mass * self.capture_fraction
        # Drivable height step between neighbouring patches, voxels: the rise
        # over one patch length at the climb angle, at least 1.
        self.climb_voxels = max(
            1, int(self.patch_x * math.tan(math.radians(self.climb_degrees)) / self.voxel)
        )
        self.reach_voxels = int(self.reach / self.voxel + 1e-9)
        self.done_voxels = self.done_tol / self.voxel
        self.progress_voxels = self.progress / self.voxel


@dataclass
class Site:
    """A grid of patches, each an integer height in voxels, plus the machine."""

    cfg: SiteCfg
    height: Grid                # current surface, voxels
    target: Grid                # final surface, voxels
    entry: Patch                # the patch the machine enters from
    dump: Patch                 # where spoil goes
    at: Patch | None = None     # machine's patch; defaults to entry
    clock: float = 0.0          # seconds elapsed
    trims: int = 0
    log: list = field(default_factory=list)     # (patch, succeeded, clock) per trim

    def __post_init__(self) -> None:
        if self.at is None:
            self.at = self.entry

    # -- queries -------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.height), len(self.height[0])

    def pending(self) -> set[Patch]:
        """Patches still above their target, by more than done_tol."""
        ny, nx = self.shape
        return {(y, x) for y in range(ny) for x in range(nx)
                if self.height[y][x] > self.target[y][x] + self.cfg.done_voxels}

    def reachable(self) -> set[Patch]:
        """Flood fill from the entry patch, stepping between neighbours only
        where the height difference is within the climb limit."""
        ny, nx = self.shape
        seen = {self.entry}
        q = deque([self.entry])
        lim = self.cfg.climb_voxels
        while q:
            iy, ix = q.popleft()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jy, jx = iy + dy, ix + dx
                if 0 <= jy < ny and 0 <= jx < nx and (jy, jx) not in seen:
                    if abs(self.height[jy][jx] - self.height[iy][ix]) <= lim:
                        seen.add((jy, jx))
                        q.append((jy, jx))
        return seen

    def actionable(self) -> list[Patch]:
        """Patches that need work and can be driven to, or dug from the ground
        beside them: the machine straddles a one-lane trench, its wheels on
        the original surface, and reaches reach_voxels below them."""
        near = self.reachable()
        lowest = -self.cfg.reach_voxels
        return sorted(p for p in self.pending() if p in near or self.height[p[0]][p[1]] > lowest)

    def failed_in_a_row(self, patch) -> int:
        """How many of the patch's latest trims failed, back to its last success."""
        n = 0
        for p, ok, _ in reversed(self.log):
            if p == patch:
                if ok:
                    break
                n += 1
        return n

    def next_trim(self) -> Patch | None:
        """The next patch to trim, layer by layer: of the actionable patches on
        the highest layer, the cheapest cycle. A patch whose last trim failed
        waits while any other can go, and is set aside after `retries` failed
        trims in a row. None when nothing is left to try."""
        options = [p for p in self.actionable() if self.failed_in_a_row(p) < self.cfg.retries]
        pool = [p for p in options if self.failed_in_a_row(p) == 0] or options
        if not pool:
            return None
        top = max(self.height[y][x] for y, x in pool)
        return min((p for p in pool if self.height[p[0]][p[1]] > top - 0.5), key=self.cycle_seconds)

    def next_level(self, patch) -> float:
        """The level a trim of patch cuts to, voxels: a layer below its
        surface, not past its target."""
        y, x = patch
        return max(self.target[y][x], self.height[y][x] - 1)

    def set_aside(self) -> list[Patch]:
        """Pending patches that failed `retries` trims in a row."""
        return sorted(p for p in self.pending() if self.failed_in_a_row(p) >= self.cfg.retries)

    def travel_seconds(self, a, b) -> float:
        c = self.cfg
        d = math.hypot((a[0] - b[0]) * c.patch_y, (a[1] - b[1]) * c.patch_x)
        return d / c.drive_speed

    def cycle_seconds(self, patch) -> float:
        """Drive to the patch, trim it, haul to the dump and discharge."""
        c = self.cfg
        return (
            self.travel_seconds(self.at, patch)
            + c.trim_seconds
            + self.travel_seconds(patch, self.dump)
            + c.dump_seconds
        )

    # -- transition ----------------------------------------------------

    def trim(self, patch, height: int | None = None) -> float:
        """Apply one trim cycle and return its seconds. height is the patch's
        height after it, as measured; by default one voxel lower. A trim that
        lowered nothing costs the same time and moved no soil. The machine
        ends at the dump."""
        cost = self.cycle_seconds(patch)
        self.clock += cost
        self.trims += 1
        y, x = patch
        before = self.height[y][x]
        self.height[y][x] = before - 1 if height is None else height
        self.at = self.dump
        self.log.append((patch, self.height[y][x] < before - self.cfg.progress_voxels, self.clock))
        return cost

    def done(self) -> bool:
        return not self.pending()

    @property
    def landed(self) -> int:
        """Trims that moved soil."""
        return sum(1 for _, ok, _ in self.log if ok)

    def expected_clock(self) -> float:
        """Clock scaled by 1 / trim_success: the expected duration when each
        trim lands with probability trim_success."""
        return self.clock / self.cfg.trim_success

    def summary(self) -> str:
        c = self.cfg
        left = len(self.pending())
        moved = self.landed * c.trim_mass
        return (
            f"{self.trims:4d} trims  {self.clock/60:6.1f} min ideal  "
            f"{self.expected_clock()/60:6.1f} min at p={c.trim_success:.2f}  "
            f"{left:4d} patches left  {moved/c.soil_density:5.2f} m3 moved  "
            f"{c.trim_seconds*self.trims/max(self.clock,1e-9):4.0%} productive"
        )


# ---------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------


def greedy_plan(site: Site) -> Site:
    """Repeatedly trim the actionable patch with the cheapest cycle. Stops when
    no pending patch is reachable."""
    while options := site.actionable():
        site.trim(min(options, key=site.cycle_seconds))
    return site


def beam_plan(site: Site, width: int = 8, horizon: int = 6) -> Site:
    """Beam search over orderings of actionable trims, `horizon` trims deep
    with the `width` cheapest per node, ranked by seconds per patch cleared.
    Commits the first trim of the best branch, then searches again."""
    while site.actionable():
        before = site.pending()
        beams = [(0.0, copy.deepcopy(site), None)]
        for _ in range(horizon):
            nxt = []
            for cost, st, first in beams:
                opts = st.actionable()
                if not opts:
                    nxt.append((cost, st, first))
                    continue
                opts.sort(key=st.cycle_seconds)
                for p in opts[:width]:
                    s2 = copy.deepcopy(st)
                    c = s2.trim(p)
                    nxt.append((cost + c, s2, first or p))
            # cheapest seconds per patch cleared
            nxt.sort(key=lambda t: t[0] / max(1, len(before - t[1].pending())))
            beams = nxt[: width * 2]
        site.trim(beams[0][2])
    return site


def trench(cfg: SiteCfg, length: float, depth: float) -> Site:
    """A one-lane trench `length` long and `depth` deep: one row of patches,
    each patch_x along the trench, dug from the ground beside it, the dump
    past its far end."""
    n = max(1, round(length / cfg.patch_x))
    d = max(1, round(depth / cfg.voxel))
    return Site(cfg=cfg, height=[[0] * n], target=[[-d] * n], entry=(0, 0), dump=(0, n + 1))


# ---------------------------------------------------------------------


def _demo() -> None:
    cfg = SiteCfg()
    ny, nx = 6, 8
    # A pit six voxels deep over the middle 4 x 4 patches.
    height = [[0] * nx for _ in range(ny)]
    target = [[0] * nx for _ in range(ny)]
    for y in range(1, 5):
        for x in range(2, 6):
            target[y][x] = -6

    print(f"patch {cfg.patch_x} x {cfg.patch_y} m, one trim = {cfg.trim_mass:.1f} kg cut, "
          f"{cfg.trim_capture:.1f} kg captured")
    print(f"the drum holds {cfg.drum_capacity:.0f} kg -> "
          f"{cfg.trim_capture/cfg.drum_capacity:.2f} loads per trim")
    print(f"climb {cfg.climb_degrees:.0f} deg over a {cfg.patch_x} m patch = "
          f"{cfg.climb_voxels} voxels of step")
    npatch = sum(1 for y in range(ny) for x in range(nx) if height[y][x] > target[y][x])
    ntrim = sum(height[y][x] - target[y][x] for y in range(ny) for x in range(nx))
    print(f"objective: {npatch} patches, {ntrim} trims of material\n")

    for name, fn in (("greedy", greedy_plan), ("beam  ", beam_plan)):
        s = Site(cfg=cfg, height=[r[:] for r in height],
                 target=[r[:] for r in target], entry=(0, 0), dump=(0, 0))
        fn(s)
        print(f"  {name}  {s.summary()}")
        if not s.done():
            print(f"          HALTED with {len(s.pending())} patches unreachable "
                  f"-- this is the case a ramp would solve")


if __name__ == "__main__":
    _demo()
