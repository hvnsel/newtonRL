"""Read a goal_trace.csv and say what the policy actually did.

    python scripts/read_trace.py goal_trace.csv
    python scripts/read_trace.py goal_trace.csv --plot traj.png

Splits the trace into episodes and reports, per episode, how far the goal was,
how much of that got closed, how long it took, and -- the number this exists
for -- the forward speed on the step `reached` fired.

Arrival is `dist < goal_dist_tol` with no velocity clause, so a machine that
crosses the ball at full speed scores the same as one that settles in it. If
the arrival speeds come back near the 1.5 m/s top speed, the policy learned to
touch the point, not to stop at it, and a handoff to a digging policy is
getting a machine that is still moving.

Needs nothing but the standard library. --plot additionally wants matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict

TOP_SPEED = 1.5         # MAX_WHEEL_SPEED * WHEEL_RADIUS
DT = 0.04               # policy step


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--plot", default=None, help="write a PNG of the xy paths here")
    p.add_argument("--env", type=int, default=0, help="which env to plot")
    return p.parse_args(argv)


def _episodes(rows: list[dict]) -> list[list[dict]]:
    """Split one env's rows wherever ep_step restarts."""
    out, cur = [], []
    prev = None
    for r in rows:
        if prev is not None and r["ep_step"] <= prev:
            if cur:
                out.append(cur)
            cur = []
        cur.append(r)
        prev = r["ep_step"]
    if cur:
        out.append(cur)
    return out


def main(argv=None) -> int:
    args = _parse(argv)
    by_env: dict[int, list[dict]] = defaultdict(list)
    with open(args.trace) as fh:
        for raw in csv.DictReader(fh):
            r = {k: (int(v) if k in ("step", "ep_step", "env", "reached", "timeout")
                     else float(v)) for k, v in raw.items()}
            by_env[r["env"]].append(r)

    if not by_env:
        print("empty trace")
        return 1

    arrival_speeds: list[float] = []
    print(f"{'env':>4} {'ep':>4} {'start m':>8} {'end m':>7} {'closed':>7} "
          f"{'steps':>6} {'path m':>7} {'v@end':>7} {'peak v':>7}  outcome")
    for env in sorted(by_env):
        for k, ep in enumerate(_episodes(by_env[env])):
            first, last = ep[0], ep[-1]
            path = sum(
                math.dist((a["x"], a["y"]), (b["x"], b["y"]))
                for a, b in zip(ep, ep[1:])
            )
            peak = max(r["speed_fwd"] for r in ep)
            outcome = "reached" if last["reached"] else ("timeout" if last["timeout"] else "cut")
            if last["reached"]:
                arrival_speeds.append(last["speed_fwd"])
            print(f"{env:>4} {k:>4} {first['dist']:>8.2f} {last['dist']:>7.2f} "
                  f"{first['dist'] - last['dist']:>7.2f} {len(ep):>6} {path:>7.2f} "
                  f"{last['speed_fwd']:>7.2f} {peak:>7.2f}  {outcome}")

    print()
    if arrival_speeds:
        n = len(arrival_speeds)
        mean = sum(arrival_speeds) / n
        print(f"arrivals: {n}")
        print(f"  speed at arrival   mean {mean:.2f}  min {min(arrival_speeds):.2f}  "
              f"max {max(arrival_speeds):.2f} m/s")
        print(f"  as a fraction of the {TOP_SPEED} m/s top speed: {mean / TOP_SPEED:.0%}")
        print()
        if mean > 0.5 * TOP_SPEED:
            print("  -> It is driving THROUGH the goal, not stopping at it. Arrival has")
            print("     no velocity condition, so this is the reward being satisfied as")
            print("     written, not the policy misbehaving.")
        else:
            print("  -> It is slowing into the goal.")
    else:
        print("no episode in this trace ended in an arrival")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print(f"\n--plot needs matplotlib; everything above needed nothing extra")
            return 0
        eps = _episodes(by_env.get(args.env, []))
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6))
        for k, ep in enumerate(eps):
            ax.plot([r["x"] for r in ep], [r["y"] for r in ep], lw=1.4,
                    label=f"ep {k} ({'reached' if ep[-1]['reached'] else 'timeout'})")
            ax.plot(ep[0]["x"], ep[0]["y"], "o", ms=5, color="black")
            ax.plot(ep[0]["goal_x"], ep[0]["goal_y"], "*", ms=16)
            bx.plot([r["ep_step"] * DT for r in ep], [r["dist"] for r in ep], lw=1.4)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"env {args.env}: path (dot = spawn, star = goal)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        bx.axhline(0.5, ls="--", lw=1, color="red")
        bx.set_xlabel("time [s]")
        bx.set_ylabel("distance to goal [m]")
        bx.set_title("distance to goal (dashed = 0.5 m tolerance)")
        bx.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nwrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
