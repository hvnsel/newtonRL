"""Read a training run's TensorBoard scalars without a browser.

    isaaclab -p scripts/peek.py                      # newest navigate run
    isaaclab -p scripts/peek.py --run <dir>           # a specific one
    isaaclab -p scripts/peek.py --task excavate
    isaaclab -p scripts/peek.py --tags reached mean_std

Batch and nohup runs redirect stdout to a file, so Python buffers it and the
log stays empty for a long time. The event file is written every iteration
regardless, so this is the way to watch a run.

Prints first, last and the final few values of every scalar, then the three
that say whether PPO is healthy at all:

  Loss/learning_rate  an adaptive schedule answering high KL by halving the
                      rate will drive this to rsl_rl's 1e-5 floor, and
                      nothing trains after that.
  Policy/mean_std     the policy's exploration width. It starts at
                      init_noise_std and should fall. One that has not left
                      its starting value is not learning, whatever the
                      reward curves do.
  Episode/reached     the task metric, last because it is the noisiest: read
                      the spread of the tail, not the last number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

HEALTH = ("Loss/learning_rate", "Policy/mean_std", "Episode/reached")


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="run directory; default is the newest")
    p.add_argument("--task", default="excavator_navigate",
                   help="log subdirectory under logs/rsl_rl")
    p.add_argument("--root", default="logs/rsl_rl")
    p.add_argument("--tail", type=int, default=4)
    p.add_argument("--tags", nargs="*", default=None, help="substrings to filter tags by")
    return p.parse_args(argv)


def _newest_run(root: Path, task: str) -> Path:
    base = root / task
    if not base.is_dir():
        raise SystemExit(f"no such log directory: {base}")
    runs = [d for d in base.iterdir() if d.is_dir() and any(d.glob("events.out.tfevents.*"))]
    if not runs:
        raise SystemExit(f"no event files under {base}")
    return max(runs, key=lambda d: max(f.stat().st_mtime for f in d.glob("events.out.tfevents.*")))


def _scalars(run: Path) -> dict[str, list[float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise SystemExit(
            "tensorboard is not importable here. Run this inside the container:\n"
            "  apptainer exec --nv -B $LUNA_SCRATCH:$LUNA_SCRATCH $SIF \\\n"
            "    /workspace/isaaclab/isaaclab.sh -p scripts/peek.py"
        )
    # 0 means keep every point rather than EventAccumulator's reservoir sample,
    # which would quietly thin the tail this script exists to read.
    acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
    acc.Reload()
    return {tag: [e.value for e in acc.Scalars(tag)] for tag in acc.Tags()["scalars"]}


def main(argv=None) -> int:
    args = _parse(argv)
    run = Path(args.run) if args.run else _newest_run(Path(args.root), args.task)
    series = _scalars(run)
    if not series:
        print(f"{run}: event file exists but holds no scalars yet")
        return 0

    iters = max(len(v) for v in series.values())
    print(f"\nrun    {run}")
    print(f"iters  {iters}\n")

    tags = sorted(series)
    if args.tags:
        tags = [t for t in tags if any(s.lower() in t.lower() for s in args.tags)]

    width = max((len(t) for t in tags), default=0)
    for tag in tags:
        v = series[tag]
        tail = ", ".join(f"{x:.4g}" for x in v[-args.tail:])
        print(f"{tag:<{width}}  first={v[0]:>10.4g}  last={v[-1]:>10.4g}  tail=[{tail}]")

    print("\n--- health ---")
    for tag in HEALTH:
        v = series.get(tag)
        if not v:
            print(f"{tag:<20}  absent")
            continue
        # Enough of a window to tell a trend from a single noisy point.
        w = v[-min(len(v), 20):]
        print(f"{tag:<20}  now={v[-1]:.4g}  start={v[0]:.4g}  "
              f"last{len(w)}: min={min(w):.4g} max={max(w):.4g} mean={sum(w)/len(w):.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
