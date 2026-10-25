"""Run NVIDIA's own two-way MPM coupling example headless and report sinkage.

    bash workstation/train.sh -p scripts/newton_twoway_check.py
    bash workstation/train.sh -p scripts/newton_twoway_check.py --seconds 6

Nothing in this file is ours. It constructs Newton's
examples/mpm/example_mpm_twoway_coupling.py unchanged, with a null viewer, and
prints where the rigid bodies come to rest.

Why: in our environment a parked machine sinks through the bed until the rigid
floor stops it, and the sinkage tracks the bed depth and nothing else --
not cohesion at 10x, not density at 4x, not the solver voxel, not any of six
coupling parameters. That is consistent with the soil having no bearing
capacity at all. This separates "Newton's implicit MPM cannot hold a load" from
"we are driving it wrong", and it does so with NVIDIA's code rather than ours.

The example drops boxes of up to 0.3 m half-extent from z = 2.0 onto a
2 x 2 x 0.5 m sand bed sitting on a ground plane. So:

    body z settling near 0.5 + hz   the sand carries them, the solver is fine
    body z settling near hz         they sank through to the floor

Its sand also differs from ours in two ways worth noting if it does hold:
particles_per_cell is 3.0 against our 2.0, and it jitters particle placement
by a full radius where we place at cell centres with jitter 0.
"""

from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--every", type=float, default=0.5, help="seconds between reports")
    args = p.parse_args()

    import numpy as np
    import newton  # noqa: F401
    from newton._src.viewer.viewer_null import ViewerNull
    from newton.examples.mpm.example_mpm_twoway_coupling import Example

    viewer = ViewerNull(num_frames=10**9)
    ex = Example(viewer, None)

    n = ex.model.body_count
    print(f"[twoway] {n} rigid bodies, sand bed 2 x 2 x 0.5 m on a ground plane")
    print("[twoway] resting ON the sand is z ~ 0.5 + half-height; "
          "through to the floor is z ~ half-height")

    frames = int(args.seconds * ex.fps)
    report = max(1, int(args.every * ex.fps))

    def heights():
        q = ex.state_0.body_q.numpy()
        return np.asarray([float(q[i][2]) for i in range(n)])

    z0 = heights()
    print(f"[twoway] t= 0.00s  z={np.round(z0, 3)}")
    for f in range(frames):
        ex.step()
        if (f + 1) % report == 0:
            z = heights()
            print(f"[twoway] t={(f + 1) / ex.fps:5.2f}s  z={np.round(z, 3)}")

    z = heights()
    print()
    print(f"[twoway] final min {z.min():.3f}  max {z.max():.3f}  mean {z.mean():.3f}")
    print("[twoway] bodies near 0.70-0.80 are floating on the sand; "
          "bodies near 0.20-0.30 went through to the floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
