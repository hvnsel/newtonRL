#!/usr/bin/env python3
"""Add PNG frame capture to an Isaac Lab MPM demo.

    apptainer exec "$SIF" cat \
      /workspace/isaaclab/scripts/demos/mpm/newton_mpm_granular.py \
      > "$LUNA_SCRATCH/mpm_frames.py"
    python3 patch_demo_frames.py "$LUNA_SCRATCH/mpm_frames.py"

    FRAME_DIR=$LUNA_SCRATCH/frames FRAME_EVERY=1 \
      apptainer exec --nv ... "$SIF" \
      /workspace/isaaclab/isaaclab.sh -p "$LUNA_SCRATCH/mpm_frames.py" \
      --max_steps 400 --device cuda:0 --viz newton_gl

Isaac Lab 3.x's video recorder is built by the environment base class and
driven by env.step(), and the demos construct a SimulationContext directly
without an environment, so they carry no --video. The recorder's frame source,
visualizer.render_rgb_array(), is exposed on every rendering visualizer, and
the demo loop already calls sim.render().

FRAME_EVERY=1 captures every step. The demos run at dt=1/100, so 400 steps is
four seconds of simulated time, a thirteen second clip at 30 fps, and takes
long enough to look like a hang.
"""

from __future__ import annotations

import sys
from pathlib import Path

HELPER = '''
def _save_frame(sim, count: int) -> None:
    """Write one PNG from whichever active visualizer can hand us pixels."""
    import os

    import numpy as np

    out = os.environ.get("FRAME_DIR", "frames")
    os.makedirs(out, exist_ok=True)
    for v in getattr(sim, "visualizers", []):
        if not hasattr(v, "render_rgb_array"):
            continue
        arr = v.render_rgb_array()
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        path = os.path.join(out, "frame_%05d.png" % count)
        try:
            import imageio.v3 as iio

            iio.imwrite(path, arr)
        except Exception:
            from PIL import Image

            Image.fromarray(arr).save(path)
        print("[FRAME] %s %s" % (path, arr.shape), flush=True)
        return
    # Names the visualizers that are active.
    active = [getattr(v.cfg, "visualizer_type", "?") for v in getattr(sim, "visualizers", [])]
    print("[FRAME] no visualizer exposed render_rgb_array; active: %s" % (active or ["none"]), flush=True)
'''

ANCHOR = "def run_simulator(sim, scene) -> None:"

OLD_TAIL = """        if sim.is_rendering:
            sim.render()
        count += 1"""

NEW_TAIL = """        if sim.is_rendering:
            sim.render()
        if count % int(os.environ.get("FRAME_EVERY", "10")) == 0:
            _save_frame(sim, count)
        count += 1"""


def patch(path: Path) -> None:
    src = path.read_text()
    if "_save_frame" in src:
        print(f"already patched: {path}")
        return
    if ANCHOR not in src:
        raise SystemExit(f"{path}: no run_simulator() -- is this an MPM demo?")
    if OLD_TAIL not in src:
        raise SystemExit(
            f"{path}: the demo loop is not shaped as expected. Re-read "
            "run_simulator() and move the _save_frame call after sim.render()."
        )
    src = src.replace(ANCHOR, HELPER.strip() + "\n\n\n" + ANCHOR, 1)
    src = src.replace(OLD_TAIL, NEW_TAIL, 1)
    if "\nimport os\n" not in src:
        src = src.replace("import math", "import math\nimport os", 1)
    path.write_text(src)
    print(f"patched {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    patch(Path(sys.argv[1]))
