"""Graphs from navigate playback traces, and the training run's curves.

    ~/IsaacLab/.venv/bin/python scripts/plot_nav.py ~/nav_media/nav_*.csv --out ~/nav_media
    ~/IsaacLab/.venv/bin/python scripts/plot_nav.py goal_trace.csv --out plots \
        --run logs/rsl_rl/excavator_navigate/<run>

workstation/nav_videos.sh runs this after recording. Each trace is one
playback run, written by goal_trace_envs, and every episode in it is drawn.
Writes, into --out:

  <trace>.png    one per trace and named after it, so nav_03.png goes with
                 nav_03.mp4: the path with the goal pose, and distance,
                 heading error and speed against time
  tracking.png   every episode's distance and heading error against time
  arrival.png    where each episode finished against the tolerances, and how
                 long it took against how far it had to go
  training.png   the run's arrival rate, end errors and curriculum by
                 iteration: --run, else the newest navigate run under logs/
  summary.csv    one row per episode

Tolerances and the episode split are read_trace.py's, and the TensorBoard
reader is peek.py's, so the three scripts agree on what an arrival is.
Needs matplotlib, which Isaac Lab's venv carries.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from peek import _newest_run, _scalars  # noqa: E402
from read_trace import DIST_TOL, DT, HEAD_TOL_DEG, SPEED_TOL, TOP_SPEED, _episodes  # noqa: E402

# Footprint drawn on the paths: drum to drum with the arms stowed (1.75 m
# either side of the chassis centre), wheel face to wheel face across.
MACHINE_LEN = 3.5
MACHINE_W = 1.35
GOAL_GREEN = "#1a9e3a"

# Iterations averaged, centred, for the training curves; the raw series is
# drawn faint behind. The goal ceiling is a staircase and is drawn raw.
SMOOTH = 25

TRAINING = (
    ("Episode/reached", "arrived and held (fraction)", ("Episode/near", "got near")),
    ("Episode/end_dist", "distance to goal at episode end [m]", None),
    ("Episode/end_heading_deg", "heading error at episode end [deg]", None),
    ("Curriculum/goal_dist_hi", "goal distance ceiling [m]", None),
    ("Curriculum/terrain_level", "mean terrain level (0-9)", None),
    ("Train/mean_reward", "mean episode return", None),
)
RAW = ("Curriculum/goal_dist_hi",)


def _parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("traces", nargs="+", help="goal_trace CSVs, one playback run each")
    p.add_argument("--out", default=".", help="directory for the figures and summary.csv")
    p.add_argument("--run", default=None,
                   help="training run directory for training.png; default is the newest")
    p.add_argument("--no-training", action="store_true", help="skip training.png")
    return p.parse_args(argv)


def _load(path: Path) -> list[list[dict]]:
    """Episodes of env 0, as read_trace parses them."""
    by_env: dict[int, list[dict]] = defaultdict(list)
    with open(path) as fh:
        for raw in csv.DictReader(fh):
            r = {k: (int(v) if k in ("step", "ep_step", "env", "reached", "timeout")
                     else float(v)) for k, v in raw.items()}
            by_env[r["env"]].append(r)
    return _episodes(by_env[min(by_env)]) if by_env else []


def _measure(name: str, ep: list[dict]) -> dict:
    """What an episode did. Errors are read on the step the hold completed,
    where navigate hands the machine over, else on the last row."""
    first, last = ep[0], ep[-1]
    arrive = next((r for r in ep if r["reached"]), None)
    at = arrive or last
    outcome = "reached" if arrive else ("timeout" if last["timeout"] else "not reached")
    path = sum(math.dist((a["x"], a["y"]), (b["x"], b["y"])) for a, b in zip(ep, ep[1:]))
    return {
        "name": name,
        "start_m": first["dist"],
        "outcome": outcome,
        "arrive_s": arrive["ep_step"] * DT if arrive else float("nan"),
        "pos_err_cm": 100.0 * at["dist"],
        "head_err_deg": abs(at["herr"]),
        "speed_mps": at["speed"],
        "path_m": path,
        "end_pos_err_cm": 100.0 * last["dist"],
        "end_head_err_deg": abs(last["herr"]),
    }


def _t(ep):
    return np.array([r["ep_step"] * DT for r in ep])


def _col(ep, key):
    return np.array([r[key] for r in ep])


def _footprint(ax, x, y, yaw, **kw):
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = 0.5 * MACHINE_LEN, 0.5 * MACHINE_W
    corners = [(x + c * dx - s * dy, y + s * dx + c * dy)
               for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
    ax.add_patch(Polygon(corners, closed=True, fill=False, **kw))
    ax.annotate("", xy=(x + c * hl, y + s * hl), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=kw.get("edgecolor", "k"), lw=1.3))


def _goal(ax, ep):
    """The goal pose. Its heading is the machine's yaw plus the heading
    error, on any row; the first is as good as the last."""
    r = ep[0]
    gx, gy = r["goal_x"], r["goal_y"]
    gyaw = r["yaw"] + math.radians(r["herr"])
    ax.add_patch(plt.Circle((gx, gy), 0.25, color=GOAL_GREEN, alpha=0.35, zorder=3))
    ax.annotate("", xy=(gx + 2.0 * math.cos(gyaw), gy + 2.0 * math.sin(gyaw)), xytext=(gx, gy),
                arrowprops=dict(arrowstyle="-|>", color=GOAL_GREEN, lw=2.2), zorder=4)
    ax.plot(gx, gy, "+", ms=12, mew=2, color=GOAL_GREEN, zorder=5)
    return gx, gy, gyaw


def _episode_figure(name: str, eps: list[list[dict]], stats: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (ax, bx), (cx, dx) = axes
    lo = np.array([np.inf, np.inf])
    hi = -lo

    for k, (ep, st) in enumerate(zip(eps, stats)):
        x, y = _col(ep, "x"), _col(ep, "y")
        t = _t(ep)
        # The path, coloured by time.
        seg = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
        lc = LineCollection(seg, cmap="viridis", lw=2.2, zorder=2)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        _footprint(ax, x[0], y[0], ep[0]["yaw"], edgecolor="0.35", lw=1.0, ls="--")
        _footprint(ax, x[-1], y[-1], ep[-1]["yaw"], edgecolor="#1f4e9e", lw=1.4)
        gx, gy, _ = _goal(ax, ep)
        ax.plot(x[0], y[0], "o", color="black", ms=6, zorder=5)
        lo = np.minimum(lo, [min(x.min(), gx), min(y.min(), gy)])
        hi = np.maximum(hi, [max(x.max(), gx), max(y.max(), gy)])

        arrive = st["arrive_s"]
        for axis in (bx, cx, dx):
            if not math.isnan(arrive):
                axis.axvline(arrive, color=GOAL_GREEN, ls="--", lw=1.2)
        bx.semilogy(t, np.maximum(_col(ep, "dist"), 1e-3), lw=1.6)
        cx.semilogy(t, np.maximum(np.abs(_col(ep, "herr")), 1e-2), lw=1.6)
        dx.plot(t, _col(ep, "speed"), lw=1.6, label="planar speed" if k == 0 else None)
        dx.plot(t, _col(ep, "speed_fwd"), lw=1.0, alpha=0.7,
                label="forward speed (signed)" if k == 0 else None)

    pad = 0.5 * MACHINE_LEN + 1.0
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("path, coloured by time  (dashed box = start, solid = end, green = goal pose)")
    fig.colorbar(lc, ax=ax, label="time [s]", shrink=0.8)

    bx.axhline(DIST_TOL, color="red", ls="--", lw=1.3, label=f"{100 * DIST_TOL:.0f} cm tolerance")
    bx.set_ylabel("distance to goal [m]")
    bx.set_title("position error")
    cx.axhline(HEAD_TOL_DEG, color="red", ls="--", lw=1.3, label=f"{HEAD_TOL_DEG:.0f} deg tolerance")
    cx.set_ylabel("|heading error| [deg]")
    cx.set_title("heading error against the goal heading")
    dx.axhline(SPEED_TOL, color="red", ls="--", lw=1.3, label=f"{SPEED_TOL} m/s settle gate")
    dx.axhline(TOP_SPEED, color="grey", ls=":", lw=1.2, label=f"{TOP_SPEED} m/s cap")
    dx.axhline(0.0, color="black", lw=0.6)
    dx.set_ylabel("speed [m/s]")
    dx.set_title("speed")
    for axis in (bx, cx, dx):
        axis.set_xlabel("time [s]")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8, loc="upper right")

    st = stats[0]
    if st["outcome"] == "reached":
        head = (f"{name}: goal {st['start_m']:.1f} m away, arrived at {st['arrive_s']:.1f} s, "
                f"{st['pos_err_cm']:.1f} cm and {st['head_err_deg']:.2f} deg off")
    else:
        head = f"{name}: goal {st['start_m']:.1f} m away, {st['outcome']} ({st['pos_err_cm'] / 100:.2f} m off)"
    if len(eps) > 1:
        head += f"  [{len(eps)} episodes]"
    fig.suptitle(head + "\n(green dashed line: the 1 s hold completed, where navigate hands over)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _tracking_figure(runs: list[tuple[str, list[dict]]], out: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.8))
    for name, ep in runs:
        t = _t(ep)
        line, = ax.semilogy(t, np.maximum(_col(ep, "dist"), 1e-3), lw=1.4, label=name)
        bx.semilogy(t, np.maximum(np.abs(_col(ep, "herr")), 1e-2), lw=1.4, color=line.get_color())
        arrive = next((r for r in ep if r["reached"]), None)
        if arrive:
            ax.plot(arrive["ep_step"] * DT, max(arrive["dist"], 1e-3), "o", ms=5, color=line.get_color())
            bx.plot(arrive["ep_step"] * DT, max(abs(arrive["herr"]), 1e-2), "o", ms=5,
                    color=line.get_color())
    ax.axhline(DIST_TOL, color="red", ls="--", lw=1.4)
    bx.axhline(HEAD_TOL_DEG, color="red", ls="--", lw=1.4)
    ax.set_ylabel("distance to goal [m]")
    ax.set_title(f"position error  (red: {100 * DIST_TOL:.0f} cm tolerance, dot: arrival)")
    bx.set_ylabel("|heading error| [deg]")
    bx.set_title(f"heading error  (red: {HEAD_TOL_DEG:.0f} deg tolerance, dot: arrival)")
    for axis in (ax, bx):
        axis.set_xlabel("time [s]")
        axis.grid(alpha=0.3, which="both")
    if len(runs) <= 16:
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(f"tracking error, {len(runs)} episodes", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _arrival_figure(stats: list[dict], out: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6))
    hit = [s for s in stats if s["outcome"] == "reached"]
    miss = [s for s in stats if s["outcome"] != "reached"]

    ax.add_patch(Rectangle((0, 0), 100 * DIST_TOL, HEAD_TOL_DEG, color=GOAL_GREEN, alpha=0.15,
                           label="arrival tolerance"))
    ax.scatter([s["pos_err_cm"] for s in hit], [s["head_err_deg"] for s in hit],
               s=50, color=GOAL_GREEN, edgecolor="black", zorder=3, label="arrival (hold complete)")
    for s in hit:
        ax.annotate(s["name"], (s["pos_err_cm"], s["head_err_deg"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlim(0, 100 * DIST_TOL * 1.4)
    ax.set_ylim(0, HEAD_TOL_DEG * 1.4)
    ax.set_xlabel("position error at arrival [cm]")
    ax.set_ylabel("|heading error| at arrival [deg]")
    ax.set_title(f"arrival accuracy: {len(hit)} of {len(stats)} episodes arrived")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    d = np.array([s["start_m"] for s in hit])
    if len(d):
        bx.scatter(d, [s["arrive_s"] for s in hit], s=50, color=GOAL_GREEN, edgecolor="black",
                   zorder=3, label="arrival")
    for s in hit:
        bx.annotate(s["name"], (s["start_m"], s["arrive_s"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    top = max([s["start_m"] for s in stats] + [1.0])
    grid = np.linspace(0, top * 1.05, 50)
    bx.plot(grid, grid / TOP_SPEED, color="grey", ls=":", label=f"straight line at {TOP_SPEED} m/s")
    for s in miss:
        bx.axvline(s["start_m"], color="red", alpha=0.4, lw=1)
    bx.set_xlabel("goal distance from spawn [m]")
    bx.set_ylabel("time to arrival [s]")
    bx.set_title("time to arrive" + (f"  (red lines: {len(miss)} not arrived)" if miss else ""))
    bx.grid(alpha=0.3)
    bx.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _smooth(v: np.ndarray, w: int) -> np.ndarray:
    """Centred mean over w points, the window shrinking at the ends, so the
    curve neither lags the raw series nor jumps at the edges."""
    c = np.cumsum(np.insert(v, 0, 0.0))
    i = np.arange(len(v))
    lo = np.clip(i - w // 2, 0, len(v))
    hi = np.clip(i + w // 2 + 1, 0, len(v))
    return (c[hi] - c[lo]) / (hi - lo)


def _training_figure(run: Path, out: Path) -> bool:
    series = _scalars(run)
    if not series:
        print(f"[training] {run} holds no scalars")
        return False
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for axis, (tag, label, extra) in zip(axes.flat, TRAINING):
        v = series.get(tag)
        if not v:
            axis.text(0.5, 0.5, f"{tag}\nnot in this run", ha="center", va="center",
                      transform=axis.transAxes)
            axis.set_axis_off()
            continue
        v = np.asarray(v, dtype=float)
        it = np.arange(len(v))
        if tag in RAW:
            axis.plot(it, v, lw=1.8, drawstyle="steps-post")
        else:
            line, = axis.plot(it, _smooth(v, SMOOTH), lw=1.8, label=tag.split("/")[-1])
            axis.plot(it, v, lw=0.6, alpha=0.3, color=line.get_color())
        if extra and series.get(extra[0]):
            w = np.asarray(series[extra[0]], dtype=float)
            axis.plot(np.arange(len(w)), _smooth(w, SMOOTH), lw=1.4, ls="--", label=extra[1])
            axis.legend(fontsize=8)
        axis.set_title(label)
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.3)
    fig.suptitle(f"training: {run.name}  ({SMOOTH}-iteration mean, raw faint)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    args = _parse(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stats: list[dict] = []
    runs: list[tuple[str, list[dict]]] = []
    wrote: list[str] = []
    for path in sorted(Path(p) for p in args.traces):
        eps = [ep for ep in _load(path) if len(ep) > 1]
        if not eps:
            print(f"  {path.name}: no rows, skipped")
            continue
        names = [path.stem] if len(eps) == 1 else [f"{path.stem}_ep{k}" for k in range(len(eps))]
        ep_stats = [_measure(n, ep) for n, ep in zip(names, eps)]
        _episode_figure(path.stem, eps, ep_stats, out / f"{path.stem}.png")
        wrote.append(f"{path.stem}.png")
        stats += ep_stats
        runs += list(zip(names, eps))

    if not stats:
        print("no episodes in any trace")
        return 1

    print(f"{'episode':<12} {'goal m':>7} {'outcome':>12} {'t s':>6} {'err cm':>7} "
          f"{'err deg':>8} {'v m/s':>6} {'path m':>7}")
    for s in stats:
        print(f"{s['name']:<12} {s['start_m']:>7.2f} {s['outcome']:>12} {s['arrive_s']:>6.1f} "
              f"{s['pos_err_cm']:>7.1f} {s['head_err_deg']:>8.2f} {s['speed_mps']:>6.2f} "
              f"{s['path_m']:>7.2f}")
    hit = [s for s in stats if s["outcome"] == "reached"]
    print(f"\narrived {len(hit)}/{len(stats)}")
    if hit:
        print(f"  position error  mean {np.mean([s['pos_err_cm'] for s in hit]):.1f}  "
              f"max {max(s['pos_err_cm'] for s in hit):.1f} cm   (tolerance {100 * DIST_TOL:.0f})")
        print(f"  heading error   mean {np.mean([s['head_err_deg'] for s in hit]):.2f}  "
              f"max {max(s['head_err_deg'] for s in hit):.2f} deg  (tolerance {HEAD_TOL_DEG:.0f})")
        print(f"  time to arrive  mean {np.mean([s['arrive_s'] for s in hit]):.1f} s")

    _tracking_figure(runs, out / "tracking.png")
    _arrival_figure(stats, out / "arrival.png")
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)
    wrote += ["tracking.png", "arrival.png", "summary.csv"]

    if not args.no_training:
        try:
            run = Path(args.run) if args.run else _newest_run(Path("logs/rsl_rl"), "excavator_navigate")
            if _training_figure(run, out / "training.png"):
                wrote.append("training.png")
        except (SystemExit, OSError, ValueError) as exc:
            print(f"[training] skipped: {exc}")

    print(f"\nwrote into {out}: {', '.join(wrote)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
