"""The stroke planner's view of a site run's ground, from the heights it saves
after every move (site_run writes <video>_cut.npz): each stroke's level, its
two halves, its target, and the move it has or why it has none, tier by tier
as plan() tries them; then the move it would make. No simulator: seconds on
the CPU.

    ~/IsaacLab/.venv/bin/python scripts/planner_state.py deep50b_video_cut.npz

The run's own log of failed moves is not in the file, so the order among the
open moves is the planner's without it.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(name: str, path: str):
    """A module by its file, without importing the package around it (and Isaac Lab)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ex = load("excavator_consts", "luna_hifi_tasks/excavator/excavator.py")
st = load("strokes", "luna_hifi_tasks/planner/strokes.py")

# As site_run sets the planner up from the site env (site_env.py, excavate_env_cfg.py).
STAGE, ARM_START = 0.6, -0.20
RING = STAGE + ex.PIVOT_X + ex.ARM_LEN * math.cos(ex.ARM_RANGE[0])

f = np.load(sys.argv[1])
rects, targets, names = f["rects"], f["targets"], [str(n) for n in f["names"]]
width = float(rects[0][3] - rects[0][2]) if names[0].startswith(("step", "bottom")) else 2.0 * ex.ROTOR_HALF_LEN
cfg = st.StrokeCfg(width=width, stride=0.5 * width, ahead=0.60, behind=0.20,
                   dig_reach=ex.PIVOT_X + ex.ARM_LEN * math.cos(ARM_START))
cfg.run_in = RING + cfg.wheelbase
strokes = []
for (x0, x1, y0, y1), t, n in zip(rects, targets, names):
    if n.startswith(("step", "bottom")) or x1 - x0 > y1 - y0:       # along x; the staircase runs +x
        strokes.append(st.Stroke(0, 0.5 * (y0 + y1), (x0, x1), n, float(t), 1 if n.startswith(("step", "bottom")) else 0))
    else:
        strokes.append(st.Stroke(1, 0.5 * (x0 + x1), (y0, y1), n, float(t), 0))
site = st.StrokeSite(cfg, strokes, float(min(targets)), st.Heights(f["z"], tuple(f["lower"]), float(f["cell"])))
h = site.heights

print(f"{sys.argv[1]}: after {int(f['moves'])} moves")
print(f"{'stroke':24s} {'level':>7s} {'halves':>15s} {'target':>7s}   (cm)")
for i, s in enumerate(strokes):
    w = 0.5 * cfg.width
    halves = [h.mean(*st.Stroke(s.axis, s.across + d, s.span).rect(*s.span, 0.5 * w)) for d in (-0.5 * w, 0.5 * w)]
    mark = "" if i in site.short() else "  done"
    print(f"{s.name:24s} {100 * site.level(i):+7.1f} {100 * halves[0]:+7.1f} {100 * halves[1]:+7.1f} "
          f"{100 * site.target_of(i):+7.1f}{mark}")

for label, kw in (("normal", {}), ("bank up to %g cm" % (100 * cfg.bank), {"rise": cfg.bank}),
                  ("footprint ignored", {"anyway": True})):
    why = []
    found = site.moves(h, why, **kw)
    print(f"\n{label}: {len(found)} open" + "".join(f"\n  open   {strokes[i].name} to {100 * v:+.1f} cm"
                                                     for (i, _, v), _ in found))
    for reason in why:
        print(f"  closed {reason}")
print(f"\nnext move: {site.next_move()}")
