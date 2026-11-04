#!/usr/bin/env python3
"""Add PNG frame capture to an Isaac Lab MPM demo, in place.

    apptainer exec "$SIF" cat \
      /workspace/isaaclab/scripts/demos/mpm/newton_mpm_granular.py \
      > "$LUNA_SCRATCH/mpm_frames.py"
    python3 patch_demo_frames.py "$LUNA_SCRATCH/mpm_frames.py"

    FRAME_DIR=$LUNA_SCRATCH/frames FRAME_EVERY=10 \
      apptainer exec --nv ... "$SIF" \
      /workspace/isaaclab/isaaclab.sh -p "$LUNA_SCRATCH/mpm_frames.py" \
      --max_steps 200 --device cuda:0 --viz newton_gl

The patched loop saves the first visualizer's frame every FRAME_EVERY steps
(default 10) to FRAME_DIR/frame_<step>.png (default ./frames).
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPER = '''
def _save_frame(sim, count: int) -> None:
    """Every FRAME_EVERY steps, write the first visualizer's frame as a PNG."""
    import os

    from PIL import Image

    if count % int(os.environ.get("FRAME_EVERY", "10")):
        return
    out = os.environ.get("FRAME_DIR", "frames")
    os.makedirs(out, exist_ok=True)
    arr = sim.visualizers[0].render_rgb_array()     # uint8 (H, W, 3)
    path = os.path.join(out, "frame_%05d.png" % count)
    Image.fromarray(arr).save(path)
    print("[FRAME] %s %s" % (path, arr.shape), flush=True)
'''

ANCHOR = "def run_simulator(sim, scene) -> None:"

OLD_TAIL = """        if sim.is_rendering:
            sim.render()
        count += 1"""

NEW_TAIL = """        if sim.is_rendering:
            sim.render()
        _save_frame(sim, count)
        count += 1"""


def patch(path: Path) -> None:
    src = path.read_text()
    if ANCHOR not in src or OLD_TAIL not in src:
        raise SystemExit(f"{path}: run_simulator() loop not found; nothing patched")
    src = src.replace(ANCHOR, HELPER.strip() + "\n\n\n" + ANCHOR, 1)
    src = src.replace(OLD_TAIL, NEW_TAIL, 1)
    path.write_text(src)
    print(f"patched {path}")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
