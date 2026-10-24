"""Rank the ways a pass can end, under the excavate mass-balance reward.

removal and unheld are functions of state, not accumulated deltas, so an
episode's total telescopes to its final value and the end state is the whole
score. That makes the reward checkable without a simulator: fix an end state,
read off the points.

What it is for: the terms have to rank honest excavation above every way of
faking it, and that ordering is not obvious from the weights. Two versions of
this reward have already been wrong in ways this file catches -- paying on
min(payload, removal) scored packing the drums above honest capture, and
measuring unheld against cut_total scored digging below the plane at +22.9
where the shipped form scores it at -1.4.

Weights mirror excavate_env_cfg. Run it after changing any of them.
"""

SOIL = 1500.0          # kg/m3, cfg.soil_density
CAP = 40.0             # kg/drum, cfg.target_load_kg
LOAD = 2 * CAP         # both normalisers
CUT_REF = 0.0432       # m3, the nominal cut: 0.9 x 1.2 m taken down one voxel
W_REMOVAL = 100.0
W_UNHELD = 120.0
W_OVERCUT = 200.0
W_SUCCESS = 40.0
SHAPE_FRAC = 0.30
KAPPA = 0.50           # cfg.capture_min_fraction


def score(above: float, below: float, payload: float, unheld_vs: str = "removal"):
    """Points for a pass that ended with `above` m3 still standing over the
    plane, a `below` m3 hole under it, and `payload` kg in the drums."""
    removal = max((CUT_REF - above) * SOIL, 0.0)
    cut_total = removal + below * SOIL
    held = min(payload, LOAD)                    # capped per the drum pair
    unheld = abs((removal if unheld_vs == "removal" else cut_total) - held)
    shape = (above + below <= SHAPE_FRAC * CUT_REF) and held >= KAPPA * cut_total
    return (
        W_REMOVAL * removal / LOAD
        - W_UNHELD * unheld / LOAD
        - W_OVERCUT * below / CUT_REF
        + (W_SUCCESS if shape else 0.0)
    ), shape


CLEARED = 0.1 * CUT_REF                          # a pass that left 10% standing
CASES = {
    # name:                      (above, below, payload kg)
    "keeps what it cuts":        (CLEARED, 0.0, 0.9 * CUT_REF * SOIL),
    "capture at measured 0.733": (CLEARED, 0.0, 0.733 * 0.9 * CUT_REF * SOIL),
    "packs the drums":           (CLEARED, 0.0, 86.0),
    "pushes the layer aside":    (CLEARED, 0.0, 0.0),
    "digs down, keeps spoil":    (CLEARED, 0.25 * CUT_REF, 74.5),
    "mines off-target":          (CUT_REF, 0.0, 36.0),
}
HONEST = ("keeps what it cuts", "capture at measured 0.733")
# Must be actively loss-making, not merely ranked below honest work. Packing
# the drums is not among them: the payload is capped at the pair's target load,
# so overfilling buys nothing and the unheld charge already prices the mass
# that did not come from the work area. It is an inefficiency, not a cheat.
CHEATS = ("pushes the layer aside", "digs down, keeps spoil", "mines off-target")


def main() -> int:
    bad = []
    for variant in ("removal", "cut_total"):
        rows = [(n, *score(*v, variant)) for n, v in CASES.items()]
        shipped = variant == "removal"
        print(f"unheld measured against {variant}"
              f"{'   <- shipped' if shipped else '   (rejected)'}")
        for n, pts, ok in sorted(rows, key=lambda r: -r[1]):
            print(f"  {n:28s} {pts:+8.2f}   success={str(ok):5s}")
        print()
        if not shipped:
            continue
        pts = dict((n, p) for n, p, _ in rows)
        floor = min(pts[n] for n in HONEST)
        for n in CASES:
            if n in HONEST:
                continue
            if pts[n] >= floor:
                bad.append(f"{n} scores {pts[n]:+.2f}, at or above honest {floor:+.2f}")
        for n in CHEATS:
            if pts[n] > 0.0:
                bad.append(f"{n} scores {pts[n]:+.2f}, which is still worth doing")

    # The rejected variant has to be rejected for a reason that survives edits.
    down_ok = score(*CASES["digs down, keeps spoil"], "cut_total")[0]
    down_shipped = score(*CASES["digs down, keeps spoil"], "removal")[0]
    if not down_ok > down_shipped:
        bad.append("cut_total no longer flatters digging down; re-derive the choice")

    for line in bad:
        print(f"FAIL  {line}")
    if bad:
        return 1
    print("every way of faking it scores below honest excavation, and below zero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
