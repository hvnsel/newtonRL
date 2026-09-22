#!/usr/bin/env python3
"""Add PNG frame capture to an Isaac Lab MPM demo.

    apptainer exec "$SIF" cat \
      /workspace/isaaclab/scripts/demos/mpm/newton_mpm_granular.py \
      > "$LUNA_SCRATCH/mpm_frames.py"
    python3 cluster/patch_demo_frames.py "$LUNA_SCRATCH/mpm_frames.py"

Why a patch and not a flag: Isaac Lab 3.x DOES ship a video recorder
(`isaaclab/envs/utils/video_recorder.py`), but it is instantiated by the
environment base class and driven by `env.step()`. The demos build a
`SimulationContext` directly and never construct an env, so the recorder is
never created and the demos have no `--video`.

They do not need one. Every visualizer that can produce pixels exposes
`render_rgb_array()`, which is the same method the recorder calls, and the
demo loop already calls `sim.render()`. So frame capture is five lines in the
right place rather than a rewrite.

Run the patched copy with a visualizer that can render:

    FRAME_DIR=$LUNA_SCRATCH/frames FRAME_EVERY=1 \
      apptainer exec --nv ... "$SIF" \
      /workspace/isaaclab/isaaclab.sh -p "$LUNA_SCRATCH/mpm_frames.py" \
      --max_steps 400 --device cuda:0 --viz newton_gl

FRAME_EVERY=1 captures every step; the demos run at dt=1/100, so 400 steps is
four seconds of simulated time, which at 30 fps plays back as a thirteen
second slow-motion clip. Rendering every step is genuinely slow -- it is what
makes a short run look like a hang -- so capture sparsely while iterating.
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
    # Naming what IS active turns "no image appeared" into a one-word fix.
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
            f"{path}: the demo loop is not shaped as expected. Upstream changed it; "
            "re-read run_simulator() and move the _save_frame call after sim.render()."
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
