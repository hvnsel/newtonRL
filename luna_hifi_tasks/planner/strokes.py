"""Stroke planner: an excavation cut as straight strokes of the drum.

A stroke is a straight lane the drum's width, along x or along y. A move cuts
one stroke down a layer at most and the dig's shallowest cut at least (less
only for the last of it), running along it either way, in passes of the dig
policy that overlap, so the cut has no seams along it; strokes side by side
overlap too. The planner keeps the site's heights (m, relative to the ground as
it was; off its grid the ground) and before each move picks which stroke, which
way along it and how deep, such that every stroke can still be finished: it
searches the order ahead on the heights it expects, and the heights measured
after each move replace them.

A move is open to the rover only if, all along the stroke, its wheel tracks
either side of the cut
  - stand within level_tol of each other (one may run down in the cut beside:
    the dig trained with its lane moved up to 0.5 m over three passes),
  - meet no step over drive_step from outside the site in,
  - leave the cut no more than rise above the higher of them (no ridges),
and the drum reaches the level from them. These only order the moves: when
a bank alone shuts every move, as where the soil a pass pushes on ahead
piles up at the next stroke's start, the rover cuts into one up to bank
high, and when they shut every move it takes the best one all the same. A
stroke short of its target is always up for another cut; one whose last
moves each failed goes after the others. The stroke with the most left to
cut goes first, so a hole goes down a layer at a time and a trench's deeper
steps lead. Strokes carrying on from one another along a line (a trench's
steps) keep a staircase all the way down: each is cut no deeper than the
next one stands, nor more than its own last step below the one before it.

Pure Python and numpy; site_env.run_strokes cuts the strokes on the RFT soil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Stroke:
    """A straight lane the drum's width, along x (axis 0) or y (axis 1), its
    centre line at `across` (a y, or an x), from span[0] to span[1] along it,
    cut down to `target` (m, below the ground; None: the site's), running
    `way` along it (1 or -1; 0: either)."""

    axis: int
    across: float
    span: tuple[float, float]
    name: str = ""
    target: float | None = None
    way: int = 0

    def point(self, along, off=0.0):
        """Env-local (x, y) `along` the axis, `off` the centre line toward +y (axis 0) or +x (axis 1)."""
        return (along, self.across + off) if self.axis == 0 else (self.across + off, along)

    def rect(self, a0: float, a1: float, half: float) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) from a0 to a1 along it, half either side of the centre line."""
        lo, hi = min(a0, a1), max(a0, a1)
        c0, c1 = self.across - half, self.across + half
        return (lo, hi, c0, c1) if self.axis == 0 else (c0, c1, lo, hi)

    def yaw(self, way: int) -> float:
        """The heading running along it toward +x or +y (way 1), or back (-1)."""
        return (0.0 if way > 0 else math.pi) if self.axis == 0 else math.copysign(0.5 * math.pi, way)


Move = tuple[int, int, float]       # stroke index, way along it (1 or -1), level, m


class Heights:
    """Heights on a grid, m relative to the ground as it was: cell (j, i) is
    centred at lower + (i + 0.5, j + 0.5) * cell. Off the grid, the ground."""

    def __init__(self, z, lower: tuple[float, float], cell: float):
        self.z, self.lower, self.cell = np.asarray(z, dtype=float), lower, cell

    def copy(self) -> Heights:
        return Heights(self.z.copy(), self.lower, self.cell)

    def at(self, x, y) -> np.ndarray:
        ny, nx = self.z.shape
        i = np.floor((np.asarray(x) - self.lower[0]) / self.cell).astype(int)
        j = np.floor((np.asarray(y) - self.lower[1]) / self.cell).astype(int)
        inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        return np.where(inside, self.z[j.clip(0, ny - 1), i.clip(0, nx - 1)], 0.0)

    def _box(self, x0: float, x1: float, y0: float, y1: float) -> tuple[slice, slice]:
        """The cells centred inside the rectangle."""
        (lx, ly), c = self.lower, self.cell
        ny, nx = self.z.shape
        i0, i1 = max(0, math.ceil((x0 - lx) / c - 0.5)), min(nx, math.floor((x1 - lx) / c - 0.5) + 1)
        j0, j1 = max(0, math.ceil((y0 - ly) / c - 0.5)), min(ny, math.floor((y1 - ly) / c - 0.5) + 1)
        return slice(j0, max(j0, j1)), slice(i0, max(i0, i1))

    def mean(self, x0: float, x1: float, y0: float, y1: float) -> float:
        cells = self.z[self._box(x0, x1, y0, y1)]
        return float(cells.mean()) if cells.size else 0.0

    def peak(self, x0: float, x1: float, y0: float, y1: float) -> float:
        cells = self.z[self._box(x0, x1, y0, y1)]
        return float(cells.max()) if cells.size else 0.0

    def cut(self, x0: float, x1: float, y0: float, y1: float, level: float) -> None:
        """Lower the rectangle to level where it stands above it."""
        box = self._box(x0, x1, y0, y1)
        self.z[box] = np.minimum(self.z[box], level)


@dataclass
class StrokeCfg:
    # --- the cut ------------------------------------------------------
    width: float = 1.6          # m the drum cuts across
    stride: float = 0.8         # m at most between side-by-side strokes' centre lines: each overlaps the next by half
    layer: float = 0.05         # m a move takes off a stroke at most ...
    least: float = 0.03         # ... and at least, unless that is all it has left: the dig's shallowest cut
    done_tol: float = 0.015     # a stroke this close to its target is done
    progress: float = 0.005     # a move that lowers its stroke less than this failed
    retries: int = 3            # failed moves in a row before a stroke goes after the others
    # --- the dig's passes, set from the env -----------------------------
    ahead: float = 0.6          # m the work area runs ahead of a pass's anchor
    behind: float = 0.2         # and behind it
    dig_reach: float = 1.665    # m from the chassis to the anchor as a pass starts
    # --- the rover's footprint ----------------------------------------
    track: float = 0.575        # m the tyres run off the stroke's centre line
    wheelbase: float = 0.7      # m the wheels stand ahead and behind the chassis
    reach: float = 0.15         # m below the wheel plane the drum digs
    pitch_lever: float = 0.7    # the wheel plane carried on to the drum
    level_tol: float = 0.18     # m the two tracks may stand apart: one in a 15 cm cut beside, and a pass's overcut
    drive_step: float = 0.18    # m the highest step a wheel takes driving in ...
    step_window: float = 0.3    # ... over this much of its track
    rise: float = 0.05          # m a stroke may stand above its higher track: no ridges ...
    bank: float = 0.15          # ... but up to this when nothing else is open: the soil passes push on ahead
    patch: float = 0.1          # m either way of a wheel's centre its height is averaged over: the tyre's contact patch
    sample: float = 0.05        # m between the points checked along a track
    run_in: float = 2.9         # m outside a stroke's end its tracks are checked from: the rear wheels, lined up to go in
    # --- the search ---------------------------------------------------
    budget: int = 2000          # moves the order search tries
    strain_bin: float = 0.03    # m: ways in this close in strain count as alike


@dataclass
class StrokeSite:
    """Strokes to cut down to target (m, below the ground), or each to its
    own, on heights."""

    cfg: StrokeCfg
    strokes: list[Stroke]
    target: float
    heights: Heights
    log: list = field(default_factory=list)     # (stroke, m it went down) per move

    # -- queries -------------------------------------------------------

    def target_of(self, i: int) -> float:
        t = self.strokes[i].target
        return self.target if t is None else t

    def halves(self, i: int, a0: float, a1: float, h: Heights | None = None) -> float:
        """The height of stroke i's lane from a0 to a1 along it, m: the higher of
        its two halves side by side. Where strokes overlap, a half is shared
        with the neighbour, so ground a neighbour has cut half of still stands
        uncut, and is cut in its turn, over the seam."""
        s, w = self.strokes[i], 0.5 * self.cfg.width
        return max((h or self.heights).mean(*Stroke(s.axis, s.across + d, s.span).rect(a0, a1, 0.5 * w))
                   for d in (-0.5 * w, 0.5 * w))

    def level(self, i: int, h: Heights | None = None) -> float:
        """Stroke i's height, m, by its halves all along it."""
        return self.halves(i, *self.strokes[i].span, h)

    def work_level(self, i: int, anchor: float, way: int, h: Heights | None = None) -> float:
        """The height of a pass's work area, m, by its halves: at its level, the pass is not needed."""
        c = self.cfg
        return self.halves(i, anchor - way * c.behind, anchor + way * c.ahead, h)

    def beside(self, i: int, j: int | None) -> bool:
        """Whether strokes i and j run the same way and overlap."""
        if j is None or i == j:
            return False
        a, b = self.strokes[i], self.strokes[j]
        return a.axis == b.axis and abs(a.across - b.across) < self.cfg.width - 1e-9

    def next_along(self, i: int, way: int) -> int | None:
        """The stroke carrying on from stroke i along its line, running `way`."""
        s = self.strokes[i]
        end = s.span[1] if way > 0 else s.span[0]
        return next((j for j, t in enumerate(self.strokes) if t.axis == s.axis and abs(t.across - s.across) < 1e-6
                     and abs((t.span[0] if way > 0 else t.span[1]) - end) < 1e-6), None)

    def failed_in_a_row(self, i: int) -> int:
        n = 0
        for j, down in reversed(self.log):
            if j == i:
                if down >= self.cfg.progress:
                    break
                n += 1
        return n

    def short(self, h: Heights | None = None) -> list[int]:
        """Strokes not yet down to their targets."""
        return [i for i in range(len(self.strokes)) if self.level(i, h) > self.target_of(i) + self.cfg.done_tol]

    def done(self) -> bool:
        return not self.short()

    def anchors(self, i: int, way: int) -> list[float]:
        """Where along stroke i its passes start, in order running `way`: the first
        work area at the stroke's near end, the last at its far end, passes no
        more than a work area's reach ahead apart, so they overlap."""
        c, (lo, hi) = self.cfg, self.strokes[i].span
        first = lo + c.behind if way > 0 else hi - c.behind
        last = hi - c.ahead if way > 0 else lo + c.ahead
        if way * (last - first) <= 0:
            return [first]
        n = math.ceil(abs(last - first) / c.ahead - 1e-9) + 1
        return [first + (last - first) * k / (n - 1) for k in range(n)]

    def work_rect(self, i: int, anchor: float, way: int) -> tuple[float, float, float, float]:
        c = self.cfg
        return self.strokes[i].rect(anchor - way * c.behind, anchor + way * c.ahead, 0.5 * c.width)

    def entry(self, i: int, way: int, out: float = 2.0) -> tuple[float, float]:
        """A point on stroke i's centre line, out beyond the end it is entered from."""
        s = self.strokes[i]
        return s.point((s.span[0] if way > 0 else s.span[1]) - way * out)

    def wheel(self, i: int, along: float, off: float, h: Heights,
              cut: tuple[float, float, float] | None = None) -> float:
        """The height under a tyre `along` stroke i, `off` its centre line: the
        mean over its contact patch; given cut (a0, a1, v), no higher than v
        from a0 to a1 along it, where the passes before have cut down to v."""
        s, p = self.strokes[i], self.cfg.patch
        z = h.mean(*Stroke(s.axis, s.across + off, s.span).rect(along - p, along + p, p))
        return min(z, cut[2]) if cut and min(cut[:2]) <= along <= max(cut[:2]) else z

    def strain(self, i: int, way: int, v: float, h: Heights, why: list | None = None,
               rise: float | None = None) -> float | None:
        """How hard cutting stroke i running `way` along it down to v is on the
        rover, on heights h: the most its wheel tracks stand apart or step, m.
        None if it cannot: a track limit broken, the drum short of v, or the cut
        a bank over rise (cfg.rise if None) above the tracks; the rule it broke
        appended to why, if given."""
        def no(reason: str) -> None:
            if why is not None:
                why.append(reason)

        c, s = self.cfg, self.strokes[i]
        anchors = self.anchors(i, way)
        chassis = [a - way * c.dig_reach for a in anchors]
        # In along the stroke from well outside it to the last pass's front wheels.
        near = s.span[0] if way > 0 else s.span[1]
        path = np.arange(near - way * c.run_in, chassis[-1] + way * (c.wheelbase + c.sample), way * c.sample)
        tracks = [h.at(*s.point(path, off)) for off in (-c.track, c.track)]
        apart = float(np.abs(tracks[0] - tracks[1]).max())
        k = max(1, round(c.step_window / c.sample))
        step = max(float(np.abs(t[k:] - t[:-k]).max()) for t in tracks) if len(path) > k else 0.0
        if apart > c.level_tol:
            return no(f"tracks {100 * apart:.0f} cm apart")
        if step > c.drive_step:
            return no(f"a {100 * step:.0f} cm step driving in")
        # At every pass the drum reaches v from where the wheels then stand, on the cut the passes
        # before made, and its work area stands no more than rise above the higher front wheel now.
        rise = c.rise if rise is None else rise
        cut = None
        for a, ch in zip(anchors, chassis):
            front, rear = ([self.wheel(i, ch + d * c.wheelbase, off, h, cut) for off in (-c.track, c.track)]
                           for d in (way, -way))
            f, r = 0.5 * sum(front), 0.5 * sum(rear)
            reach = f + c.pitch_lever * (f - r) - c.reach
            if v < reach - 1e-9:
                return no(f"the drum reaches {100 * reach:.0f} cm, not {100 * v:.0f}")
            now = max(self.wheel(i, ch + way * c.wheelbase, off, h) for off in (-c.track, c.track))
            bank = h.mean(*self.work_rect(i, a, way)) - now
            if bank > rise:
                return no(f"a {100 * bank:.0f} cm bank above the front wheels")
            cut = (anchors[0] - way * c.behind, a + way * c.ahead, v)
        return max(apart, step)

    def moves(self, h: Heights, why: list | None = None, rise: float | None = None,
              anyway: bool = False) -> list[tuple[Move, float]]:
        """The open moves on h, each with its strain: every stroke to come a
        layer down, not past the target, its way along it or either, and at
        least `least` unless that is all it has left. In a staircase a step is
        cut no deeper than the next one stands while that one is short, so no
        step stands above the one before it, not even with the soil a pass
        pushes on ahead; and no more than its own last step below the one
        before it, so the way down stays one step at a time. Given why, each
        closed one's reason is appended to it; rise as in strain. With anyway,
        the rover's footprint (strain) shuts none."""
        c, out = self.cfg, []
        short = self.short(h)
        for i in short:
            s, lv, t = self.strokes[i], self.level(i, h), self.target_of(i)
            for way in (s.way,) if s.way else (1, -1):
                label = f"{s.name} {'+' if way > 0 else '-'}{'xy'[s.axis]}"
                v, held = max(t, lv - c.layer), ""
                before = self.next_along(i, -way)
                if before in short:
                    lowest = self.level(before, h) - (self.target_of(before) - t)
                    if lowest > v:
                        v, held = lowest, f"a step at most below {self.strokes[before].name}"
                j = self.next_along(i, way)
                if j in short and self.level(j, h) > v:
                    v, held = self.level(j, h), f"waits for {self.strokes[j].name}"
                if v > lv - min(c.least, lv - t) + 1e-9:
                    if why is not None:
                        why.append(f"{label}: {held}")
                    continue
                reasons = [] if why is not None else None
                strain = 0.0 if anyway else self.strain(i, way, v, h, reasons, rise)
                if strain is not None:
                    out.append(((i, way, v), strain))
                elif why is not None:
                    why.append(f"{label}: {reasons[0]}")
        return out

    def apply(self, move: Move, h: Heights) -> None:
        """The move on h as planned: the stroke's lane down to its level."""
        i, _, v = move
        s = self.strokes[i]
        h.cut(*s.rect(*s.span, 0.5 * self.cfg.width), v)

    # -- the plan ------------------------------------------------------

    def plan(self, at: tuple[float, float] | None = None) -> list[Move] | None:
        """An order of moves that finishes every stroke on the heights expected.
        Depth first: the stroke with the most left to cut first (to the
        nearest done_tol), so a hole goes down a layer at a time across it and
        a trench's deeper steps lead; then one overlapping the stroke cut last,
        so a layer sweeps across the strokes side by side, each cut over half
        of the one before; then the one with the deepest target; then the way
        along it gentlest on the rover (its strain to the nearest strain_bin);
        then, for the first move, the way in nearest `at`. A stroke whose last
        `retries` moves each failed goes after the others. If none is found
        within budget, the best first move alone. If a bank alone shuts every
        move, the best one cutting into a bank up to cfg.bank; if the rover's
        footprint shuts every move, the best one all the same: a stroke short
        of its target is always up for another cut. None once every one is done."""
        order: list[Move] = []
        tried = 0
        before = self.log[-1][0] if self.log else None

        def ranked(h: Heights, first: bool, rise: float | None = None, anyway: bool = False) -> list[Move]:
            last = order[-1][0] if order else before

            def key(ms: tuple[Move, float]):
                (i, way, _), strain = ms
                near = math.dist(at, self.entry(i, way)) if first and at else 0.0
                left = self.level(i, h) - self.target_of(i)
                return (self.failed_in_a_row(i) >= self.cfg.retries, -round(left / self.cfg.done_tol),
                        not self.beside(i, last), self.target_of(i), round(strain / self.cfg.strain_bin), near)
            return [m for m, _ in sorted(self.moves(h, rise=rise, anyway=anyway), key=key)]

        def search(h: Heights, first: bool) -> bool:
            nonlocal tried
            if not self.short(h):
                return True
            for m in ranked(h, first):
                tried += 1
                if tried > self.cfg.budget:
                    return False
                nxt = h.copy()
                self.apply(m, nxt)
                order.append(m)
                if search(nxt, False):
                    return True
                order.pop()
            return False

        top = ranked(self.heights, True)
        if not top:
            return (ranked(self.heights, True, self.cfg.bank) or ranked(self.heights, True, anyway=True))[:1] or None
        return order if search(self.heights, True) else top[:1]

    def next_move(self, at: tuple[float, float] | None = None) -> Move | None:
        order = self.plan(at)
        return order[0] if order else None

    def record(self, i: int, before: float, after: float) -> None:
        """Stroke i went from before to after (m) in a move."""
        self.log.append((i, before - after))

    def save(self, path: str) -> None:
        """The heights, the moves made and each stroke's rectangle, target and
        name, to an .npz for scripts/render_cut.py."""
        h, half = self.heights, 0.5 * self.cfg.width
        np.savez_compressed(path, z=h.z.astype(np.float32), lower=np.asarray(h.lower), cell=h.cell, moves=len(self.log),
                            rects=np.array([s.rect(*s.span, half) for s in self.strokes]),
                            targets=np.array([self.target_of(i) for i in range(len(self.strokes))]),
                            names=np.array([s.name for s in self.strokes]))


def side_by_side(cfg: StrokeCfg, across: float) -> list[float]:
    """Centre lines of strokes side by side over a strip `across` wide, m in
    from its edge: the outer two flush with its edges, the rest evenly between,
    no more than stride apart, so each overlaps the next."""
    if across <= cfg.width:
        return [0.5 * across]
    n = math.ceil((across - cfg.width) / cfg.stride - 1e-9) + 1
    return [0.5 * cfg.width + k * (across - cfg.width) / (n - 1) for k in range(n)]


def l_strokes(cfg: StrokeCfg, corner: tuple[float, float], size: float, thickness: float) -> list[Stroke]:
    """An L `size` on a side, its arms `thickness` across, its outer corner at
    `corner` (its least x and y): one arm along +x, the other up +y from it,
    each strokes side by side along it; the arm along x takes the corner."""
    x0, y0 = corner
    centres = side_by_side(cfg, thickness)
    n = len(centres)
    return ([Stroke(0, y0 + c, (x0, x0 + size), f"along x, {k + 1} of {n}") for k, c in enumerate(centres)]
            + [Stroke(1, x0 + c, (y0 + thickness, y0 + size), f"along y, {k + 1} of {n}") for k, c in enumerate(centres)])


def l_rects(corner: tuple[float, float], size: float, thickness: float) -> list[tuple[float, float, float, float]]:
    """The L's two arms as (x0, x1, y0, y1)."""
    x0, y0 = corner
    return [(x0, x0 + size, y0, y0 + thickness), (x0, x0 + thickness, y0, y0 + size)]


def hole_strokes(cfg: StrokeCfg, corner: tuple[float, float], size: float) -> list[Stroke]:
    """A square hole `size` on a side, its corner at `corner` (its least x and
    y): strokes side by side along x across it, and the same along y, the
    planner picking which way each layer goes."""
    x0, y0 = corner
    centres = side_by_side(cfg, size)
    n = len(centres)
    return ([Stroke(0, y0 + c, (x0, x0 + size), f"along x, {k + 1} of {n}") for k, c in enumerate(centres)]
            + [Stroke(1, x0 + c, (y0, y0 + size), f"along y, {k + 1} of {n}") for k, c in enumerate(centres)])


def hole_rects(corner: tuple[float, float], size: float) -> list[tuple[float, float, float, float]]:
    """The hole as (x0, x1, y0, y1)."""
    x0, y0 = corner
    return [(x0, x0 + size, y0, y0 + size)]


# A trench down a staircase: each step STEP_RISE below the one before it over
# STEP_RUN, the wheelbase, so the wheels stand at most one step apart; then a
# flat bottom BOTTOM long at the full depth. From one step the drum reaches the
# next with room to spare (StrokeCfg.reach). Wider than the drum, it is cut in
# lanes side by side, each overlapping the next, each its own staircase.
STEP_RISE, STEP_RUN, BOTTOM = 0.12, 1.4, 2.0


def trench_length(depth: float) -> float:
    """How long the staircase trench down to depth (m) is, m."""
    return (math.ceil(depth / STEP_RISE - 1e-9) - 1) * STEP_RUN + BOTTOM


def trench_strokes(cfg: StrokeCfg, corner: tuple[float, float], depth: float, width: float) -> list[Stroke]:
    """A trench `depth` deep and `width` wide along +x from `corner` (its least
    x and y): in each lane a stroke a step, each down to its own level, the last
    the bottom, all cut running down the staircase (+x)."""
    x0, y0 = corner
    n = math.ceil(depth / STEP_RISE - 1e-9)
    ends = [x0 + k * STEP_RUN for k in range(n)] + [x0 + trench_length(depth)]
    lanes = side_by_side(cfg, width)
    return [Stroke(0, y0 + c, (ends[k], ends[k + 1]),
                   (f"step {k + 1} of {n - 1}" if k < n - 1 else "bottom") + (f", lane {j + 1}" if len(lanes) > 1 else ""),
                   -min(depth, (k + 1) * STEP_RISE), way=1)
            for j, c in enumerate(lanes) for k in range(n)]


def trench_rects(corner: tuple[float, float], depth: float, width: float) -> list[tuple[float, float, float, float]]:
    """The trench as (x0, x1, y0, y1)."""
    x0, y0 = corner
    return [(x0, x0 + trench_length(depth), y0, y0 + width)]
