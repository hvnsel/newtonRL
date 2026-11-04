"""Print a training run's TensorBoard scalars: first, last and the final few
values of each, then learning rate, policy std and arrival rate over the last
20 iterations. Reads the event file directly; needs only the standard library.

    python scripts/peek.py                              # newest navigate run
    python scripts/peek.py --run <dir>
    python scripts/peek.py --task excavator_excavate
    python scripts/peek.py --tags reached mean_std
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

HEALTH = ("Loss/learning_rate", "Policy/mean_std", "Episode/reached")


def _parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="run directory; default is the newest")
    p.add_argument("--task", default="excavator_navigate", help="log subdirectory under --root")
    p.add_argument("--root", default="logs/rsl_rl")
    p.add_argument("--tail", type=int, default=4)
    p.add_argument("--tags", nargs="*", default=None, help="substrings to filter tags by")
    return p.parse_args(argv)


def _newest_run(root: Path, task: str) -> Path:
    """The run directory under root/task whose event file was written last."""
    runs = [d for d in (root / task).iterdir() if any(d.glob("events.out.tfevents.*"))]
    return max(runs, key=lambda d: max(f.stat().st_mtime for f in d.glob("events.out.tfevents.*")))


# Event file format: records of [uint64 length][uint32 crc][payload][uint32 crc].
# The payload is an Event protobuf: field 5 is a Summary, whose field 1 is a
# repeated Value with the tag in field 1 and a float simple_value in field 2.


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    out = shift = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, i
        shift += 7


def _fields(buf: bytes):
    """Yield (field_number, raw) for each field of a protobuf message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        else:
            v, i = buf[i:i + 4], i + 4
        yield fn, v


def _records(blob: bytes):
    """Yield each complete record's payload."""
    i = 0
    while i + 12 <= len(blob):
        length = int.from_bytes(blob[i:i + 8], "little")
        start = i + 12
        if start + length + 4 > len(blob):
            return                                  # record still being written
        yield blob[start:start + length]
        i = start + length + 4


def _scalars(run: Path) -> dict[str, list[float]]:
    """Every scalar series in the run, by tag, in step order."""
    series: dict[str, list[float]] = {}
    for path in sorted(run.glob("events.out.tfevents.*")):
        for payload in _records(path.read_bytes()):
            for fn, summary in _fields(payload):
                if fn != 5:
                    continue
                for sfn, value in _fields(summary):
                    if sfn != 1:
                        continue
                    tag = val = None
                    for vfn, raw in _fields(value):
                        if vfn == 1:
                            tag = raw.decode()
                        elif vfn == 2:
                            (val,) = struct.unpack("<f", raw)
                    if val is not None:
                        series.setdefault(tag, []).append(val)
    return series


def main(argv=None) -> int:
    args = _parse(argv)
    run = Path(args.run) if args.run else _newest_run(Path(args.root), args.task)
    series = _scalars(run)
    if not series:
        print(f"{run}: no scalars yet")
        return 0

    print(f"\nrun    {run}")
    print(f"iters  {max(len(v) for v in series.values())}\n")

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
        w = v[-20:]
        print(f"{tag:<20}  now={v[-1]:.4g}  start={v[0]:.4g}  "
              f"last{len(w)}: min={min(w):.4g} max={max(w):.4g} mean={sum(w)/len(w):.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
