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
DIST_TOL = 0.5          # cfg.goal_dist_tol
SPEED_TOL = 0.25        # cfg.goal_speed_tol
DT = 0.04               # policy step


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--plot", default=None, help="write a PNG of the xy paths here")
    p.add_argument("--env", type=int, default=0, help="which env to plot")
    p.add_argument("--episodes", type=int, nargs="*", default=None,
                   help="episode indices to plot; default is all of them")
    p.add_argument("--frame", choices=("goal", "terrain"), default="goal",
                   help="goal: every path normalised so the setpoint is the origin, "
                        "which makes episodes comparable. terrain: absolute env "
                        "coordinates, which shows where the goals actually were.")
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
            # Traces written before `speed` existed only carry body-frame x.
            r.setdefault("speed", abs(r["speed_fwd"]))
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
                # signed, so the reverse split below can still be counted
                arrival_speeds.append(last["speed_fwd"])
            print(f"{env:>4} {k:>4} {first['dist']:>8.2f} {last['dist']:>7.2f} "
                  f"{first['dist'] - last['dist']:>7.2f} {len(ep):>6} {path:>7.2f} "
                  f"{last['speed_fwd']:>7.2f} {peak:>7.2f}  {outcome}")

    print()
    if arrival_speeds:
        n = len(arrival_speeds)
        # speed_fwd is body-frame x and therefore SIGNED: reversing reads
        # negative. Averaging it directly lets a machine arriving backwards at
        # 1.5 cancel one arriving forwards at 1.5 and reports a sedate 0.0.
        # Magnitude is the only thing that answers "was it still moving".
        mag = [abs(v) for v in arrival_speeds]
        mean = sum(mag) / n
        reverse = [v for v in arrival_speeds if v < 0]
        over = [v for v in mag if v > TOP_SPEED]
        print(f"arrivals: {n}")
        print(f"  |speed| at arrival  mean {mean:.2f}  min {min(mag):.2f}  max {max(mag):.2f} m/s")
        print(f"  against the {TOP_SPEED} m/s commanded cap: {mean / TOP_SPEED:.0%}")
        print()
        if mean > 0.5 * TOP_SPEED:
            print("  -> Driving THROUGH the goal, not stopping at it. goal_reached has no")
            print("     velocity clause, so this is the reward satisfied as written.")
        else:
            print("  -> Slowing into the goal.")
        if reverse:
            rm = sum(abs(v) for v in reverse) / len(reverse)
            print()
            print(f"  -> {len(reverse)}/{n} ({len(reverse)/n:.0%}) arrive IN REVERSE, mean {rm:.2f} m/s.")
            print("     bearing_alignment clamps forward_speed to [0, 1], so reversing")
            print("     forgoes that reward but is never penalised, and nothing else in")
            print("     the reward refers to heading. Backwards is near-equally optimal.")
        if over:
            print()
            print(f"  -> {len(over)}/{n} ({len(over)/n:.0%}) exceed the {TOP_SPEED} m/s the wheels can")
            print(f"     command, up to {max(mag):.2f}. The drive is a velocity servo, so the")
            print("     body is outrunning its own wheels: sliding, or running downhill")
            print("     under lunar gravity. Check wheel_slip if this matters.")
    else:
        print("no episode in this trace ended in an arrival")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n--plot needs matplotlib:")
            print("  uv pip install --python <isaaclab>/.venv/bin/python matplotlib")
            print("Everything above needed nothing extra.")
            return 0

        eps = _episodes(by_env.get(args.env, []))
        if args.episodes:
            eps = [eps[i] for i in args.episodes if i < len(eps)]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        (ax, bx), (cx, dx) = axes

        # 1. the path. In the goal's frame every episode overlays and is
        #    comparable; in the terrain frame you see where the goals actually
        #    were, which is the only way to tell a policy that favours a
        #    direction from bearings that simply came up similar.
        terrain = args.frame == "terrain"
        for k, ep in enumerate(eps):
            ox, oy = (0.0, 0.0) if terrain else (ep[0]["goal_x"], ep[0]["goal_y"])
            hit = ep[-1]["reached"]
            ax.plot([r["x"] - ox for r in ep], [r["y"] - oy for r in ep],
                    lw=1.3, alpha=0.85 if hit else 0.5,
                    ls="-" if hit else "--",
                    label=f"ep {k}" + ("" if hit else " (timeout)"))
            ax.plot(ep[0]["x"] - ox, ep[0]["y"] - oy, "o", ms=5, color="black")
            ax.plot(ep[-1]["x"] - ox, ep[-1]["y"] - oy,
                    "x" if not hit else "*", ms=10 if hit else 7)
            if terrain:
                ax.add_patch(plt.Circle((ep[0]["goal_x"], ep[0]["goal_y"]), DIST_TOL,
                                        fill=False, ls="--", color="red", lw=1.2))
                ax.plot(ep[0]["goal_x"], ep[0]["goal_y"], "+", ms=11, color="red", mew=1.8)
        if not terrain:
            ax.add_patch(plt.Circle((0, 0), DIST_TOL, fill=False, ls="--", color="red", lw=1.5))
            ax.plot(0, 0, "+", ms=14, color="red", mew=2)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]" if terrain else "x relative to goal [m]")
        ax.set_ylabel("y [m]" if terrain else "y relative to goal [m]")
        ax.set_title(
            "path in the env frame  (+ = each goal, dot = spawn)" if terrain
            else f"path in the goal frame  (+ = setpoint, dashed = {DIST_TOL} m tolerance)"
        )
        ax.grid(alpha=0.3)
        if len(eps) <= 8:
            ax.legend(fontsize=8, loc="best")

        # 2. the error signal the policy is actually closing
        for ep in eps:
            bx.plot([r["ep_step"] * DT for r in ep], [r["dist"] for r in ep], lw=1.3,
                    alpha=0.85 if ep[-1]["reached"] else 0.45)
        bx.axhline(DIST_TOL, ls="--", lw=1.5, color="red")
        bx.set_xlabel("time [s]")
        bx.set_ylabel("distance to setpoint [m]")
        bx.set_title("tracking error")
        bx.grid(alpha=0.3)

        # 3. speed against the gate. Planar magnitude, which is what
        #    goal_reached tests; body-frame x would read negative in reverse.
        for ep in eps:
            cx.plot([r["ep_step"] * DT for r in ep], [r["speed"] for r in ep], lw=1.3,
                    alpha=0.85 if ep[-1]["reached"] else 0.45)
        cx.axhline(SPEED_TOL, ls="--", lw=1.5, color="red")
        cx.axhline(TOP_SPEED, ls=":", lw=1.2, color="grey")
        cx.set_xlabel("time [s]")
        cx.set_ylabel("planar speed [m/s]")
        cx.set_title(f"speed  (dashed = {SPEED_TOL} m/s gate, dotted = {TOP_SPEED} m/s cap)")
        cx.grid(alpha=0.3)

        # 4. where it is pointing versus where the goal is. Near 180 means it
        #    is driving the traverse in reverse, which this machine is built to
        #    do -- a drum at each end.
        for ep in eps:
            err = []
            for r in ep:
                want = math.atan2(r["goal_y"] - r["y"], r["goal_x"] - r["x"])
                e = (want - r["yaw"] + math.pi) % (2 * math.pi) - math.pi
                err.append(abs(math.degrees(e)))
            dx.plot([r["ep_step"] * DT for r in ep], err, lw=1.3,
                    alpha=0.85 if ep[-1]["reached"] else 0.45)
        dx.axhline(90, ls=":", lw=1.2, color="grey")
        # A reverse traverse sits at exactly 180, which ylim(0, 180) draws on
        # the top spine and hides -- the one thing this panel exists to show.
        dx.axhline(180, ls="--", lw=1.2, color="red", alpha=0.6)
        dx.set_ylim(-6, 192)
        dx.set_yticks([0, 45, 90, 135, 180])
        dx.set_xlabel("time [s]")
        dx.set_ylabel("|bearing to setpoint| [deg]")
        dx.set_title("heading error  (near 0 = nose first, near 180 = reversing)")
        dx.grid(alpha=0.3)

        fig.suptitle(f"env {args.env}: {len(eps)} episode(s) against the setpoint", fontsize=13)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nwrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
