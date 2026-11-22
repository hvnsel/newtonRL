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
Move = tuple[Patch, int]     # a patch and the way the rover faces: 1 along +x, -1 along -x
Trim = tuple[Patch, int, float]     # a move and the level it cuts to, voxels


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

    min_cut: float = 0.03
    """The shallowest cut a pass makes, metres: the dig is commanded 3-8 cm
    below the patch's surface."""

    # --- the rover's footprint, for trims on the soil -------------------
    wheel_patches: tuple[int, ...] = (1, 2, 3)
    """Patches behind the cut, along the pass, that the rover's wheels stand
    on: the chassis is 1.67 m behind the drum and the wheels 0.7 m either side
    of it, on the lanes either side of the cut (the drum is 0.95 m wide, the
    tyres run 0.475-0.675 m off the rover's centre line)."""

    level_tol: float = 0.025
    """How far apart in height the two wheel tracks may stand to dig, metres."""

    drive_step: float = 0.10
    """Highest step the wheels take driving in, along a track or across the
    two, metres."""

    pitch_lever: float = 0.7
    """The wheel plane carried on to the drum: 0.97 m ahead of the front wheels
    over a 1.4 m wheelbase."""

    search_budget: int = 3000
    """Trims the order search tries before it settles for the best single trim."""

    overcut: float = 0.01
    """How much deeper than asked a pass may cut, metres, asked at least
    min_cut: the order search takes the trims that strand nothing even then
    first."""

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
        self.min_cut_voxels = self.min_cut / self.voxel
        self.level_voxels = self.level_tol / self.voxel
        self.drive_voxels = self.drive_step / self.voxel
        self.overcut_voxels = self.overcut / self.voxel


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

    # -- the rover on the soil ------------------------------------------
    #
    # Rows are lanes across the site, columns patches along it; the rover
    # drives in from the west (column -1) along the lane it cuts, its wheel
    # tracks on the lanes either side. Off the grid is the ground as it was.

    def level_at(self, y: int, x: int) -> float:
        """Height of patch (y, x), voxels; 0 off the grid."""
        ny, nx = self.shape
        return self.height[y][x] if 0 <= y < ny and 0 <= x < nx else 0.0

    def lowest_at(self, y: int, x: int, aside=()) -> float:
        """The lowest patch (y, x) will go: its target while it has trims to
        come, else where it is."""
        ny, nx = self.shape
        if not (0 <= y < ny and 0 <= x < nx):
            return 0.0
        h, t = self.height[y][x], self.target[y][x]
        return t if h > t + self.cfg.done_voxels and (y, x) not in aside else h

    def wheel_columns(self, move: Move) -> list[int]:
        """Columns the wheels stand on to cut the patch, nearest the cut first."""
        (_, x), facing = move[:2]
        return [x - facing * k for k in self.cfg.wheel_patches]

    def path_columns(self, move: Move) -> range:
        """Columns the wheels cross driving in from the west to the pass start."""
        return range(-1, max(self.wheel_columns(move)) + 1)

    def reachable_from_west(self, move: Move) -> bool:
        """A pit is entered down its ramp only, so the rover cannot stand past
        its east end; a one-lane trench it straddles from end to end."""
        ny, nx = self.shape
        return ny == 1 or max(self.wheel_columns(move)) < nx

    def levels(self, patch) -> list[float]:
        """Levels a trim of patch may cut to, voxels: a layer down, not past its
        target, or less, to the level of a lane two over in its column, the
        other track whenever this one carries a wheel, so the two can stand
        level."""
        y, x = patch
        h = self.height[y][x]
        full = self.next_level(patch)
        pair = {self.level_at(y - 2, x), self.level_at(y + 2, x)}
        return [full] + sorted(v for v in pair if full < v < h - self.cfg.progress_voxels)

    def reach(self, move: Move) -> float | None:
        """The deepest level the drum gets to for the move, voxels, or None if
        the rover cannot get there: it drives in with no step over drive_step
        along or across its tracks, stands with them level within level_tol,
        and the drum reaches reach_voxels below the wheel plane at the middle
        of the tracks."""
        c = self.cfg
        (y, x), _ = move[:2]
        if not self.reachable_from_west(move):
            return None
        lv = self.level_at
        path = list(self.path_columns(move))
        for a, b in zip(path, path[1:]):
            if max(abs(lv(y - 1, b) - lv(y - 1, a)), abs(lv(y + 1, b) - lv(y + 1, a)), abs(lv(y - 1, b) - lv(y + 1, b))) > c.drive_voxels:
                return None
        wheels = self.wheel_columns(move)
        if any(abs(lv(y - 1, a) - lv(y + 1, a)) > c.level_voxels for a in wheels):
            return None
        front = 0.5 * (lv(y - 1, wheels[0]) + lv(y + 1, wheels[0]))
        rear = 0.5 * (lv(y - 1, wheels[-1]) + lv(y + 1, wheels[-1]))
        return front + c.pitch_lever * (front - rear) - c.reach_voxels

    def can_trim(self, trim: Trim) -> bool:
        """The rover can make the cut now: it gets there and the drum gets to the level."""
        deepest = self.reach(trim)
        return deepest is not None and trim[2] >= deepest - 1e-9

    def could_trim(self, move: Move, aside=()) -> bool:
        """Whether the cut could still come one day: heights only fall, so two
        patches whose floors or targets already hold them apart, across the
        tracks or along one, never come together again."""
        c = self.cfg
        (y, x), _ = move
        if not self.reachable_from_west(move):
            return False
        wheels = set(self.wheel_columns(move))

        def apart(a: tuple[int, int], b: tuple[int, int]) -> float:
            """How close the two patches can still come, voxels."""
            return max(0.0, self.lowest_at(*a, aside) - self.level_at(*b), self.lowest_at(*b, aside) - self.level_at(*a))

        path = list(self.path_columns(move))
        for a in path:
            if apart((y - 1, a), (y + 1, a)) > (c.level_voxels if a in wheels else c.drive_voxels):
                return False
        return all(apart((row, a), (row, b)) <= c.drive_voxels for row in (y - 1, y + 1) for a, b in zip(path, path[1:]))

    def stranded(self, aside=()) -> bool:
        """Whether some patch with trims to come can never be cut, either way round."""
        return any(
            not (self.could_trim((p, 1), aside) or self.could_trim((p, -1), aside))
            for p in self.pending() if p not in aside
        )

    def moves(self, aside=(), facing: int | None = None, later=()) -> list[Trim]:
        """The trims the rover can make now, facing `facing` if given: the
        higher patches first, then the cheaper cycle, then facing +x, then the
        full layer; patches in `later` last."""
        out = []
        for p in self.pending():
            if p in aside:
                continue
            y, x = p
            for f in (facing,) if facing else (1, -1):
                deepest = self.reach((p, f))
                if deepest is None:
                    continue
                # A layer down, the paired track's level, or as deep as the drum
                # gets; a cut no shallower than a pass makes, unless it finishes.
                h, done = self.height[y][x], self.target[y][x] + self.cfg.done_voxels
                for v in dict.fromkeys(self.levels(p) + [max(self.next_level(p), deepest)]):
                    if v >= deepest - 1e-9 and (v <= h - self.cfg.min_cut_voxels + 1e-9 or v <= done):
                        out.append((p, f, v))
        out.sort(key=lambda m: (m[0] in later, -self.height[m[0][0]][m[0][1]], self.cycle_seconds(m[0]), -m[1], m[2]))
        return out

    def plan(self, facing: int | None = None) -> list[Trim] | None:
        """An order of trims that finishes the site, each feasible when it comes
        and none stranding a patch; the first faces `facing` if given, and
        avoids a patch whose last trim failed if it can. Depth first: the trims
        that strand nothing even if the pass cuts deeper than asked first, then
        the one that leaves the most cuts possible. If none is found within
        search_budget trims tried, that best first trim alone; None if there
        is no trim that strands nothing."""
        work = copy.copy(self)
        work.height = [row[:] for row in self.height]
        aside = set(self.set_aside())
        later = {p for p in self.pending() if self.failed_in_a_row(p)}
        order: list[Trim] = []
        tried = 0

        def options() -> int:
            """Cuts, each patch either way round, that could still come one day."""
            return sum(work.could_trim((p, f), aside) for p in work.pending() if p not in aside for f in (1, -1))

        def ranked(first: bool) -> list[Trim]:
            """Each trim that strands nothing: those that strand nothing cut at
            least min_cut and overcut past that first, then the one leaving the
            most cuts possible."""
            out = []
            for i, (p, f, v) in enumerate(work.moves(aside, facing if first else None, later if first else ())):
                y, x = p
                before = work.height[y][x]
                work.height[y][x] = v
                if not work.stranded(aside):
                    n = options()
                    work.height[y][x] = min(v, before - self.cfg.min_cut_voxels) - self.cfg.overcut_voxels
                    out.append((work.stranded(aside), -n, i, (p, f, v)))
                work.height[y][x] = before
            return [t for *_, t in sorted(out)]

        def search(first: bool) -> bool:
            nonlocal tried
            if not work.pending() - aside:
                return True
            for p, f, v in (top if first else ranked(False)):
                tried += 1
                if tried > self.cfg.search_budget:
                    return False
                y, x = p
                before = work.height[y][x]
                work.height[y][x] = v
                order.append((p, f, v))
                if search(False):
                    return True
                order.pop()
                work.height[y][x] = before
            return False

        top = ranked(True)
        if not top:
            return None
        # No full order within the budget: the trim that leaves the most cuts possible.
        return order if search(True) else top[:1]

    def next_trim(self, facing: int) -> tuple[Patch, float] | None:
        """The next patch to cut, the rover facing `facing`, and the level to
        cut it to: the first trim of plan(facing). None if it can cut nothing
        facing this way without stranding a patch."""
        order = self.plan(facing)
        return (order[0][0], order[0][2]) if order else None

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
    each patch_x along the trench, dug from the ground beside it; the way in
    and the dump at its west end."""
    n = max(1, round(length / cfg.patch_x))
    d = max(1, round(depth / cfg.voxel))
    return Site(cfg=cfg, height=[[0] * n], target=[[-d] * n], entry=(0, 0), dump=(0, -2))


def pit(cfg: SiteCfg, length: float, depth: float, step: float, lanes: int = 5) -> Site:
    """A pit `depth` deep, `lanes` lanes wide and `length` long, entered down
    a staircase at its west end: each patch along it `step` deeper than the
    last, down to the floor. The lane along each side wall is cut with one
    wheel on the ground outside the pit, so it goes no deeper than the drum
    reaches below the wheels: a shelf."""
    n = max(1, round(length / cfg.patch_x))
    d, s, shelf = depth / cfg.voxel, step / cfg.voxel, float(cfg.reach_voxels)

    def lane(deepest: float) -> list[float]:
        return [-min(deepest, s * (x + 1)) for x in range(n)]

    target = [lane(shelf)] + [lane(d) for _ in range(lanes - 2)] + [lane(shelf)]
    return Site(cfg=cfg, height=[[0.0] * n for _ in range(lanes)], target=target,
                entry=(lanes // 2, 0), dump=(lanes // 2, -2))


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
