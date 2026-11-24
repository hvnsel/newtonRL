"""Join a site run's video clips into one mp4, during the run or after it.

    ~/IsaacLab/.venv/bin/python scripts/join_clips.py deep50d_video

With --video, site_run's recorder writes a clip into that directory every
simulated minute (site_run.CLIP_STEPS). This joins the clips there so far, in
order, into <directory>_so_far.mp4; a clip still being written is left out.
site_run joins them all into <directory>.mp4 when the run ends.

The clips are copied end to end by ffmpeg's concat demuxer, not decoded: one
ffmpeg process, its memory flat however many clips there are (a run of hours
has hundreds), and seconds to run.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg


def complete(path: Path) -> bool:
    """Whether an mp4 is written out: its top-level boxes run exactly to the
    end of the file, and one is its index (moov). A clip still being written
    has neither."""
    try:
        size, at, kinds = path.stat().st_size, 0, set()
        with path.open("rb") as f:
            while at < size:
                f.seek(at)
                head = f.read(16)
                if len(head) < 8:
                    return False
                n, kind = int.from_bytes(head[:4], "big"), head[4:8]
                if n == 1:                      # a 64-bit size follows
                    if len(head) < 16:
                        return False
                    n = int.from_bytes(head[8:16], "big")
                elif n == 0:                    # runs to the end of the file
                    n = size - at
                if n < 8:
                    return False
                kinds.add(kind)
                at += n
        return at == size and b"moov" in kinds
    except OSError:
        return False


def join(paths, out: str) -> int:
    """The clips at paths, in order, into out, leaving out any not yet
    written out. How many went in."""
    paths = [Path(p) for p in paths]
    done = [p for p in paths if complete(p)]
    for p in paths:
        if p not in done:
            print(f"[join] left out {p}: still being written")
    if not done:
        return 0
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in done:
            quoted = str(p.resolve()).replace("'", "'\\''")
            f.write(f"file '{quoted}'\n")
        listing = f.name
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", listing, "-c", "copy", out], check=True)
    finally:
        Path(listing).unlink(missing_ok=True)
    return len(done)


if __name__ == "__main__":
    folder = sys.argv[1].rstrip("/")
    out = f"{folder}_so_far.mp4"
    print(f"[join] {out}: {join(sorted(Path(folder).glob('clip_*.mp4')), out)} clips")
