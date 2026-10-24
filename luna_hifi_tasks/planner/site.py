"""Heightfield site model for the excavation planner.

The planner's unit of action is one trim: the dig policy clears one voxel of
depth from a patch the size of its own footprint. Nothing below that matters
to it, so the site is a grid of patches each carrying an integer height in
voxels, and a trim is `height -= 1`. That makes every transition exact
arithmetic and the whole model costs nothing to simulate -- no particles, no
rigid bodies, no Isaac Lab.

What it is for: deciding the order of trims, where to put the dump pile, and
not stranding the machine below a face it cannot climb. Travel dominates --
at a 10 m dump distance only about a third of the machine's time is spent
cutting -- so the ordering is most of the value.

Every constant below is measured, and says where from. The two that are
guesses are marked as such.

Validate against MPM, do not trust this alone: it cannot know whether the dig
policy's success rate holds when trimming beside a face rather than in open
ground, nor whether the navigate policy can cross terrain it has itself
modified while being blind to it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

Patch = tuple[int, int]      # (iy, ix)
Grid = list[list[int]]       # heights in voxels; no numpy, the grids are tiny


@dataclass
class SiteCfg:
    # --- geometry -----------------------------------------------------
    voxel: float = 0.04
    """Metres per voxel. The dig policy's cut depth is exactly one of these,
    because particles sit one per cell and nothing thinner is representable."""

    patch_x: float = 0.90
    patch_y: float = 1.20
    """The trim footprint: 0.60 m of work area ahead of the anchor plus 0.30
    behind, by the 1.2 m drum swath. Sized so one pass can finish one layer."""

    # --- what a trim does ---------------------------------------------
    soil_density: float = 1500.0
    """kg/m3. Apollo surface regolith."""

    capture_fraction: float = 0.733
    """Of the soil a trim disturbs, the share reaching the bore rather than
    jamming in the shell. Measured across the calibration sweep, range
    0.70-0.77 over a 5x span of advance speed and 4.5x of depth."""

    drum_pair_capacity: float = 52.0
    """kg the two drums hold at normal packing. Measured peak bore load. One
    trim captures 47.5 kg, so the machine dumps after essentially every trim
    -- which is what makes this a routing problem."""

    trim_seconds: float = 13.0
    """Cutting time for one trim. The drum is throughput limited at 4-7 kg/s
    almost regardless of depth or advance speed, and one layer is 65 kg."""

    trim_success: float = 0.5
    """Probability a trim meets its shape tolerance. Taken from the dig
    policy's Episode/shape_done. Feeds the planner one-for-one: at 0.5 the
    cycle count doubles, so raising it is worth more than planner cleverness.
    Update this from the training run before trusting any plan's duration."""

    # --- travel -------------------------------------------------------
    drive_speed: float = 1.0
    """m/s while hauling. GUESS. Navigate's cap is 1.5 m/s and it arrived at
    0.22, so the cruise figure wants measuring from a goal trace."""

    dump_seconds: float = 3.0
    """Time to discharge. Measured: both drums emptied 99.6% in about 2.5 s
    of reversed drum."""

    # --- reachability -------------------------------------------------
    climb_degrees: float = 10.0
    """The slope the laden machine can drive. GUESS, and the single most
    important number here, because it is what makes a ramp necessary. Soil
    friction of 0.84 gives a 40 degree soil-on-soil limit and the tipping
    limit is 60, but wheel traction on loose regolith with 52 kg aboard is
    much less. Measure it before relying on any plan that digs deep."""

    def __post_init__(self) -> None:
        self.patch_area = self.patch_x * self.patch_y
        self.trim_mass = self.patch_area * self.voxel * self.soil_density
        self.trim_capture = self.trim_mass * self.capture_fraction
        # Voxels of height difference between neighbouring patches that the
        # machine can still drive. A patch length of run at the climb angle.
        self.climb_voxels = max(
            1, int(self.patch_x * math.tan(math.radians(self.climb_degrees)) / self.voxel)
        )


@dataclass
class Site:
    """A grid of patches, each an integer height in voxels, plus the machine."""

    cfg: SiteCfg
    height: Grid                # current surface, voxels
    target: Grid                # where the surface should end up, voxels
    entry: Patch                # the patch the machine enters from
    dump: Patch                 # where spoil goes
    at: Patch | None = None     # machine's patch; defaults to entry
    load: float = 0.0           # kg in the drums
    clock: float = 0.0          # seconds elapsed
    trims: int = 0
    log: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.at is None:
            self.at = self.entry

    # -- queries -------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.height), len(self.height[0])

    def pending(self) -> set[Patch]:
        """Patches still above their target."""
        ny, nx = self.shape
        return {(y, x) for y in range(ny) for x in range(nx)
                if self.height[y][x] > self.target[y][x]}

    def reachable(self) -> set[Patch]:
        """Flood fill from the entry patch, stepping between neighbours only
        where the height difference is within the climb limit.

        This is the constraint that keeps the machine from digging itself in,
        and it is exact -- no learning needed to enforce it. What it does NOT
        give you is a ramp: filtering to reachable patches stops you getting
        stuck but leaves you halting early, because building access means
        trimming patches you do not want excavated, before you need them.
        That is what lookahead is for.
        """
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
        """Patches that need work and can be driven to."""
        return sorted(self.pending() & self.reachable())

    def travel_seconds(self, a, b) -> float:
        c = self.cfg
        d = math.hypot((a[0] - b[0]) * c.patch_y, (a[1] - b[1]) * c.patch_x)
        return d / c.drive_speed

    def cycle_seconds(self, patch) -> float:
        """Drive to the patch, trim it, haul to the dump, discharge, and be
        back at the dump ready for the next one."""
        c = self.cfg
        return (
            self.travel_seconds(self.at, patch)
            + c.trim_seconds
            + self.travel_seconds(patch, self.dump)
            + c.dump_seconds
        )

    # -- transition ----------------------------------------------------

    def trim(self, patch, succeeded: bool = True) -> float:
        """Apply one trim. Returns the seconds it cost.

        A failed trim costs the same time and moves no soil. The baselines
        never pass succeeded=False -- they plan the ideal schedule -- so
        trim_success enters only through `expected_clock`. Pass it explicitly
        when replaying a plan against sampled outcomes.
        """
        cost = self.cycle_seconds(patch)
        self.clock += cost
        self.trims += 1
        if succeeded:
            self.height[patch[0]][patch[1]] -= 1
            self.load = min(self.load + self.cfg.trim_capture, self.cfg.drum_pair_capacity)
        self.load = 0.0                      # discharged at the dump each cycle
        self.at = self.dump
        self.log.append((patch, succeeded, self.clock))
        return cost

    def done(self) -> bool:
        return not self.pending()

    @property
    def landed(self) -> int:
        """Trims that actually moved soil, per the log."""
        return sum(1 for _, ok, _ in self.log if ok)

    def expected_clock(self) -> float:
        """Clock corrected for the dig policy's success rate.

        `self.clock` is the perfect-execution duration: the baselines below
        assume every trim lands, because a search cannot plan around a coin
        flip it has not flipped yet. A failed trim costs a full cycle and
        moves nothing, so clearing one voxel costs 1/trim_success cycles in
        expectation and the whole schedule stretches by that factor. At the
        measured 0.5 this doubles every duration, which is large enough that
        quoting the raw clock would be misleading.
        """
        return self.clock / max(self.cfg.trim_success, 1e-9)

    def summary(self) -> str:
        c = self.cfg
        left = len(self.pending())
        moved = self.landed * c.trim_mass          # what the log says moved
        return (
            f"{self.trims:4d} trims  {self.clock/60:6.1f} min ideal  "
            f"{self.expected_clock()/60:6.1f} min at p={c.trim_success:.2f}  "
            f"{left:4d} patches left  {moved/c.soil_density:5.2f} m3 moved  "
            f"{c.trim_seconds*self.trims/max(self.clock,1e-9):4.0%} productive"
        )


# ---------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------


def greedy_plan(site: Site, max_trims: int = 10_000) -> Site:
    """Always trim the reachable pending patch that costs the least to reach.

    Correct on reachability and usually decent on travel, but it will halt the
    moment every pending patch is cut off rather than build a way in.
    """
    while not site.done() and site.trims < max_trims:
        options = site.actionable()
        if not options:
            break
        best = min(options, key=site.cycle_seconds)
        site.trim(best, succeeded=True)
    return site


def beam_plan(site: Site, width: int = 8, horizon: int = 6, max_trims: int = 10_000) -> Site:
    """Short-horizon lookahead over trim orderings.

    This is where a ramp can appear: a branch that spends two or three trims
    on patches it does not need, because doing so opens patches it does. The
    horizon has to exceed the ramp's length for that to be findable -- a
    0.25 m face at 10 degrees is about 1.6 patch lengths, so a horizon of 6
    is ample at that depth and nowhere near enough for a 2 m pit.
    """
    import copy

    while not site.done() and site.trims < max_trims:
        if not site.actionable():
            break
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
                    c = s2.trim(p, succeeded=True)
                    nxt.append((cost + c, s2, first if first else p))
            # cheapest seconds per patch actually cleared
            before = site.pending()
            nxt.sort(key=lambda t: t[0] / max(1, len(before - t[1].pending())))
            beams = nxt[: width * 2]
        step = beams[0][2]
        if step is None:
            break
        site.trim(step, succeeded=True)
    return site


# ---------------------------------------------------------------------


def _demo() -> None:
    cfg = SiteCfg()
    ny, nx = 6, 8
    # A pit: take the middle down six voxels, which is the whole 0.25 m bed.
    height = [[0] * nx for _ in range(ny)]
    target = [[0] * nx for _ in range(ny)]
    for y in range(1, 5):
        for x in range(2, 6):
            target[y][x] = -6

    print(f"patch {cfg.patch_x} x {cfg.patch_y} m, one trim = {cfg.trim_mass:.1f} kg cut, "
          f"{cfg.trim_capture:.1f} kg captured")
    print(f"drums hold {cfg.drum_pair_capacity:.0f} kg -> "
          f"{cfg.trim_capture/cfg.drum_pair_capacity:.2f} loads per trim")
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
