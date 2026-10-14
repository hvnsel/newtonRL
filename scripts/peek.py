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


# ---------------------------------------------------------------------------
# Reading the event file
#
# The preferred path is tensorboard's own EventAccumulator. It is not
# importable on a login node, where apptainer does not exist either, so the
# fallback below parses the file directly: the scalars are all this script
# wants, and the format they sit in is small.
#
# A record is a little-endian uint64 length, a masked CRC32C of those eight
# bytes, the payload, then a masked CRC32C of the payload. The payload is an
# Event message whose field 2 is the step and field 5 a Summary, whose field 1
# is a repeated Value, whose field 1 is the tag and field 2 a 32-bit
# simple_value.
# ---------------------------------------------------------------------------

_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ (0x82F63B78 if _c & 1 else 0)   # CRC32C, reflected
    _CRC_TABLE.append(_c)


def _masked_crc32c(buf: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in buf:
        crc = _CRC_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    crc ^= 0xFFFFFFFF
    return ((((crc >> 15) | (crc << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF


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
    """Walk a protobuf message, yielding (field_number, wire_type, raw)."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v, i = buf[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif wt == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unknown protobuf wire type {wt}")
        yield fn, wt, v


def _records(blob: bytes):
    """Yield each record's payload, stopping at a half-written tail."""
    i, n = 0, len(blob)
    while i + 12 <= n:
        header = blob[i:i + 8]
        length = int.from_bytes(header, "little")
        hcrc = int.from_bytes(blob[i + 8:i + 12], "little")
        if _masked_crc32c(header) != hcrc:
            raise ValueError(f"record header CRC mismatch at byte {i}")
        start = i + 12
        if start + length + 4 > n:
            return      # the trainer is mid-write and this record is incomplete
        payload = blob[start:start + length]
        pcrc = int.from_bytes(blob[start + length:start + length + 4], "little")
        if _masked_crc32c(payload) != pcrc:
            raise ValueError(f"record payload CRC mismatch at byte {start}")
        yield payload
        i = start + length + 4


def _scalars_raw(run: Path) -> dict[str, list[float]]:
    import math
    import struct
    series: dict[str, list[float]] = {}
    for path in sorted(run.glob("events.out.tfevents.*")):
        for payload in _records(path.read_bytes()):
            for fn, _wt, raw in _fields(payload):
                if fn != 5:                             # Event.summary
                    continue
                for sfn, _swt, sraw in _fields(raw):
                    if sfn != 1:                        # Summary.value
                        continue
                    tag = val = None
                    for vfn, _vwt, vraw in _fields(sraw):
                        if vfn == 1:
                            tag = vraw.decode("utf-8", "replace")
                        elif vfn == 2:
                            (val,) = struct.unpack("<f", vraw)
                    if tag is None or val is None:
                        continue
                    # A misaligned walk yields unprintable tags and junk floats
                    # rather than failing outright, so refuse them here instead
                    # of printing a plausible-looking table of nonsense.
                    if not tag.isprintable() or not math.isfinite(val):
                        raise ValueError(f"implausible scalar: tag={tag!r} value={val!r}")
                    series.setdefault(tag, []).append(val)
    return series


def _scalars(run: Path) -> dict[str, list[float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return _scalars_raw(run)
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
