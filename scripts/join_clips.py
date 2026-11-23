"""Join a site run's video clips into one mp4, during the run or after it.

    ~/IsaacLab/.venv/bin/python scripts/join_clips.py deep60e_video

With --video, site_run's recorder writes a clip into that directory every
simulated minute (site_run.CLIP_STEPS). This joins the clips there so far, in
order, into <directory>_so_far.mp4; a clip still being written is left out.
site_run joins them all into <directory>.mp4 when the run ends.
"""

import sys
from pathlib import Path

from moviepy.editor import VideoFileClip, concatenate_videoclips


def join(paths, out: str) -> int:
    """The clips at paths, in order, into out, leaving out any that do not read
    yet (still being written). How many went in."""
    clips = []
    for p in paths:
        try:
            clips.append(VideoFileClip(str(p)))
        except OSError:
            print(f"[join] left out {p}: still being written")
    if clips:
        concatenate_videoclips(clips).write_videofile(out, codec="libx264", audio=False, logger=None)
    return len(clips)


if __name__ == "__main__":
    folder = sys.argv[1].rstrip("/")
    out = f"{folder}_so_far.mp4"
    print(f"[join] {out}: {join(sorted(Path(folder).glob('clip_*.mp4')), out)} clips")
