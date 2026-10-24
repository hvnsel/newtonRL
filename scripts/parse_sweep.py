#!/usr/bin/env python3
"""Reduce a dig_sweep run to the quantities an analytical soil tier needs.

Usage:  python3 scripts/parse_sweep.py ~/dig_sweep [--csv out.csv]

Reads each run's stdout and reports, per (cut depth, advance speed):

  stall_steps   how often the stall term fired, which is the draft exceeding
                available traction. The depth where this leaves zero is the
                cliff an analytical resistance model has to put in the same
                place.
  bore_f/r      peak load in each drum, kg
  wall_f/r      peak stuck in the shell and never reaching the bore, kg
  capture       bore / (bore + wall), the fraction of disturbed soil the drum
                actually takes -- the capture efficiency to fit
  kg_per_m      bore mass per metre advanced, which with the cut cross-section
                gives the removal rate
  tip_adv       vane tip speed over advance speed. Below about 1 the drum is
                dragged rather than cutting.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys

TIP_SPEED = 2.9 * 0.185          # rad/s * vane tip radius, m/s
WHEEL = 5.0 * 0.30               # MAX_WHEEL_SPEED * WHEEL_RADIUS, m/s
T_DRIVE = 7.0

PAIR = r"tensor\(\[\[\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)\s*\]\]"


def parse(path: pathlib.Path) -> dict | None:
    text = path.read_text(errors="replace")
    if "=== result ===" not in text:
        return None
    m = re.search(r"cut([0-9.]+)_drive([0-9.]+)", path.name)
    if not m:
        return None
    cut, drive = float(m.group(1)), float(m.group(2))

    bore = re.search(r"peak fill per drum.*?" + PAIR, text, re.S)
    wall = re.search(r"wall band.*?" + PAIR, text, re.S)
    if not (bore and wall):
        return None
    bf, br = float(bore.group(1)), float(bore.group(2))
    wf, wr = float(wall.group(1)), float(wall.group(2))

    secs = re.search(r"--seconds[= ]([0-9.]+)", text)
    dur = float(secs.group(1)) if secs else 25.0
    thr = re.search(r"->\s*([0-9]+)\s*env-steps/s", text)
    arm = re.search(r"to cut [0-9.]+ m -> arm ([+-][0-9.]+) rad", text)

    # stall only appears in a step's reward breakdown when it fires
    stall = len(re.findall(r"\bstall [+-][0-9.]+", text))
    adv = drive * WHEEL
    dist = max(adv * (dur - T_DRIVE), 1e-9)
    moved = bf + br + wf + wr

    return {
        "cut": cut, "drive": drive, "adv": adv,
        "tip_adv": TIP_SPEED / adv if adv > 0 else math.inf,
        "arm": float(arm.group(1)) if arm else float("nan"),
        "stall_steps": stall,
        "bore_f": bf, "bore_r": br, "wall_f": wf, "wall_r": wr,
        "capture": (bf + br) / moved if moved > 0 else float("nan"),
        "kg_per_m": (bf + br) / dist,
        "dist_m": dist,
        "steps_s": int(thr.group(1)) if thr else -1,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--csv")
    a = ap.parse_args()

    logs = sorted(pathlib.Path(a.logdir).glob("*.log"))
    rows = [r for r in (parse(p) for p in logs) if r]
    if not rows:
        print(f"no completed runs in {a.logdir}", file=sys.stderr)
        return 1

    cols = ["cut", "adv", "tip_adv", "arm", "stall_steps", "bore_f", "bore_r",
            "wall_f", "wall_r", "capture", "kg_per_m", "dist_m", "steps_s"]
    w = {c: max(len(c), 7) for c in cols}
    print("  ".join(c.rjust(w[c]) for c in cols))
    for r in sorted(rows, key=lambda r: (r["drive"], r["cut"])):
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, int):
                cells.append(f"{v}".rjust(w[c]))
            elif v != v or v == math.inf:
                cells.append("-".rjust(w[c]))
            else:
                cells.append(f"{v:.3f}".rjust(w[c]))
        print("  ".join(cells))

    print()
    depth = sorted((r for r in rows if abs(r["drive"] - 0.10) < 1e-9),
                   key=lambda r: r["cut"])
    if depth:
        clean = [r["cut"] for r in depth if r["stall_steps"] == 0]
        stalled = [r["cut"] for r in depth if r["stall_steps"] > 0]
        print("A: resistance against depth, at advance 0.15 m/s")
        print(f"   no stall up to  {max(clean):.2f} m" if clean else "   stalls at every depth tested")
        if stalled:
            print(f"   stalls from     {min(stalled):.2f} m")
            print(f"   -> put the analytical traction cliff between "
                  f"{max([c for c in clean if c < min(stalled)], default=0):.2f} and {min(stalled):.2f} m")

    speed = sorted((r for r in rows if abs(r["cut"] - 0.06) < 1e-9),
                   key=lambda r: r["adv"])
    if len(speed) > 1:
        print()
        print("B: capture against advance, at cut 0.06 m")
        for r in speed:
            print(f"   adv {r['adv']:.3f} m/s  tip/adv {r['tip_adv']:5.2f}  "
                  f"capture {r['capture']:.2f}  kg/m {r['kg_per_m']:6.2f}  "
                  f"stall {r['stall_steps']}")
        print("   capture collapsing as tip/adv approaches 1 is the plow threshold.")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
