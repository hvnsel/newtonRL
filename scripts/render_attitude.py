#!/usr/bin/env python3
"""Draw the machine's ATTITUDE over time from a play_dig dump.

scripts/render_dig.py draws the soil from above, which cannot show a machine
tipping, and the Newton backend returns no pixels from env.render(), so neither
existing view could show what this rover has actually been doing: pitching
nose-up until its rear wheels reach the bed floor while both front wheels lift
clear of the soil.

This reads the per-body world positions that play_dig now dumps and draws two
elevations per frame -- side (x-z) and rear (y-z) -- with the soil surface and
the bed floor marked, so the pitch and roll are visible directly.

    ~/IsaacLab/.venv/bin/python scripts/play_dig.py --park --probe --no_window --seconds 12
    python3 scripts/render_attitude.py dig_bed.npz attitude.gif

Falls back to a PNG strip of evenly spaced frames when matplotlib is absent.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    if "body_pos" not in d or d["body_pos"].size == 0:
        sys.exit(
            f"{path} has no body_pos. Re-run play_dig.py with --dump after pulling "
            "the commit that records it; older dumps only hold the heightmap and "
            "the chassis pose, neither of which shows attitude."
        )
    try:
        names = [str(n) for n in d["body_names"]]
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"could not read body_names from {path}: {exc!r}\n"
            "An older dump stored them as a pickled object array, which numpy "
            "1.x cannot read when numpy 2.x wrote it. The body positions "
            "themselves are fine -- re-run this script with the venv "
            "interpreter, which has the matching numpy:\n"
            "  ~/IsaacLab/.venv/bin/python scripts/render_attitude.py "
            f"{path} out.gif"
        )
    return d["body_pos"], names, d, float(d["wheel_radius"]) if "wheel_radius" in d else 0.30


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".attitude.gif")
    pos, names, d, r = load(src)
    nf = pos.shape[0]

    wheels = [i for i, n in enumerate(names) if "wheel" in n]
    drums = [i for i, n in enumerate(names) if "drum" in n or "shroud" in n]
    chassis = [i for i, n in enumerate(names) if n == "chassis"]
    bed_top = float(d["voxel"]) * 0.5 + 0.25 if "voxel" in d else 0.27
    floor = 0.02

    print(f"{nf} frames, {len(names)} bodies")
    print(f"wheels: {[names[i] for i in wheels]}")

    def ix(tag: str):
        for i in wheels:
            if tag in names[i]:
                return i
        return None

    fl, fr, rl, rr = (ix("fl"), ix("fr"), ix("rl"), ix("rr"))

    def attitude(f: int):
        """yaw, pitch, roll in degrees, from the wheel positions alone."""
        P = pos[f]
        fwd = 0.5 * (P[fl] + P[fr]) - 0.5 * (P[rl] + P[rr])   # rear axle -> front
        lat = 0.5 * (P[fl] + P[rl]) - 0.5 * (P[fr] + P[rr])   # right -> left
        yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
        pitch = math.degrees(math.atan2(fwd[2], math.hypot(fwd[0], fwd[1])))
        roll = math.degrees(math.atan2(lat[2], math.hypot(lat[0], lat[1])))
        return yaw, pitch, roll

    print()
    print("  frame      yaw    pitch     roll   wheel z spread")
    print("  (a parked machine with no wheel command should hold all three near 0)")
    for f in sorted({0, nf // 8, nf // 4, nf // 2, 3 * nf // 4, nf - 1}):
        y, pi, ro = attitude(f)
        zs = pos[f, wheels, 2]
        print(f"  {f:5d}  {y:+7.2f}  {pi:+7.2f}  {ro:+7.2f}   "
              f"{1000 * (zs.max() - zs.min()):7.0f} mm")
    print()
    for lbl, f in (("first", 0), ("last", nf - 1)):
        zs = {names[i]: pos[f, i, 2] for i in wheels}
        spread = max(zs.values()) - min(zs.values())
        print(f"  {lbl:5s} frame: wheel z " +
              "  ".join(f"{k.replace('_body','')}={v:+.4f}" for k, v in zs.items()) +
              f"   spread {1000 * spread:.0f} mm")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib unavailable ({exc!r}); writing nothing. The printed "
              f"wheel z above is the number that matters.")
        return 0

    fig, (axs, axr, axp) = plt.subplots(1, 3, figsize=(15, 4.4))

    def draw(f: int) -> None:
        for ax, (hi, vi, title, hlab) in (
            (axs, (0, 2, "side elevation (looking along -y)", "x [m]")),
            (axr, (1, 2, "rear elevation (looking along +x)", "y [m]")),
        ):
            ax.clear()
            ax.axhline(bed_top, color="#b8860b", lw=1.2, label="soil surface")
            ax.axhline(floor, color="#444", lw=1.6, label="bed floor")
            for i in wheels:
                ax.add_patch(plt.Circle((pos[f, i, hi], pos[f, i, vi]), r,
                                        fill=False, color="#1f77b4", lw=1.6))
            if drums:
                ax.scatter(pos[f, drums, hi], pos[f, drums, vi], s=26,
                           color="#d62728", zorder=3)
            if chassis:
                ax.scatter(pos[f, chassis, hi], pos[f, chassis, vi], s=60,
                           marker="s", color="#333", zorder=3)
            ax.set_xlim(-2.2, 2.6)
            ax.set_ylim(-0.15, 1.75)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(hlab, fontsize=8)
            ax.set_ylabel("z [m]", fontsize=8)
            ax.grid(alpha=0.25, lw=0.5)
        # Plan view. The side and rear elevations both HIDE yaw: a yawed
        # machine shows four separated wheels in each, which reads as the
        # wheels being in the wrong place rather than the body being turned.
        axp.clear()
        order = [i for i in (fl, fr, rr, rl) if i is not None]
        if len(order) == 4:
            poly = np.array([[pos[f, i, 0], pos[f, i, 1]] for i in order + [order[0]]])
            axp.plot(poly[:, 0], poly[:, 1], color="#1f77b4", lw=1.6)
        axp.scatter(pos[f, wheels, 0], pos[f, wheels, 1], s=40, color="#1f77b4", zorder=3)
        for i in wheels:
            axp.annotate(names[i].replace("wheel_", "").replace("_body", ""),
                         (pos[f, i, 0], pos[f, i, 1]), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
        if drums:
            axp.scatter(pos[f, drums, 0], pos[f, drums, 1], s=26, color="#d62728", zorder=3)
        axp.set_xlim(-2.2, 2.6)
        axp.set_ylim(-2.4, 2.4)
        axp.set_aspect("equal")
        axp.set_title("plan view (looking down) -- this one shows yaw", fontsize=9)
        axp.set_xlabel("x [m]", fontsize=8)
        axp.set_ylabel("y [m]", fontsize=8)
        axp.grid(alpha=0.25, lw=0.5)

        zs = pos[f, wheels, 2]
        y, pi, ro = attitude(f)
        axs.legend(loc="upper right", fontsize=7)
        fig.suptitle(f"frame {f + 1}/{nf}    yaw {y:+.1f}   pitch {pi:+.1f}   "
                     f"roll {ro:+.1f} deg    wheel z spread "
                     f"{1000 * (zs.max() - zs.min()):.0f} mm", fontsize=10)

    writer = PillowWriter(fps=12)
    with writer.saving(fig, str(out), dpi=110):
        for f in range(nf):
            draw(f)
            writer.grab_frame()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
