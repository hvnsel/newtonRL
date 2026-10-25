#!/usr/bin/env python3
"""Port Newton 1.7.0's colour-buffer capacity fix into the installed solver.

Why this exists. With strain_basis="pic8" the solver raises, during
SolverImplicitMPM construction:

    RuntimeError: Keys and values array storage must be large enough to
                  contain 2*count elements
      newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py
        _compute_coloring -> wp.utils.radix_sort_pairs

_compute_coloring sorts one entry per PARTITION CELL, but allocate_temporaries
sizes color_indices from strain_node_count, which for a particle strain basis
counts PARTICLES. Cells without particles make the cell count exceed it, the
radix sort needs 2*count, and the buffer is only 2*strain_node_count. A grid
strain basis like the P0 default hides this because its node count tracks
cells, which is why this only shows up on pic*.

Newton 1.7.0 fixes it by taking the larger of the two:

    partition_cell_count = self._strain_space_restriction.space_partition.geo_partition.cell_count()
    color_block_capacity = max(strain_node_count, partition_cell_count)

This workstation runs a hybrid install -- Newton 1.6.1 with 1.7.0's
implicit_mpm/ grafted in MINUS solver_implicit_mpm.py, which was dropped
because 1.7.0's copy calls validate_observables on a base class 1.6.1 does not
have. The colour-capacity fix is in the file that was dropped, so it is
missing. This applies just that change.

Idempotent, backs the file up once, and verifies afterwards. Reverse with
  --revert   (restores the .bak)

Usage:
  ~/IsaacLab/.venv/bin/python scripts/patch_newton_mpm.py
  ~/IsaacLab/.venv/bin/python scripts/patch_newton_mpm.py --revert
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ANCHOR = (
    "        if max_colors > 0:\n"
    "            self.color_indices = fem.borrow_temporary("
    "temporary_store, shape=(2, strain_node_count), dtype=int)\n"
)

PATCHED = (
    "        if max_colors > 0:\n"
    "            # Cell-based coloring sorts one entry per partition cell, and cells without\n"
    "            # particles make that count exceed the particle-based strain node count.\n"
    "            partition_cell_count = "
    "self._strain_space_restriction.space_partition.geo_partition.cell_count()\n"
    "            color_block_capacity = max(strain_node_count, partition_cell_count)\n"
    "            self.color_indices = fem.borrow_temporary("
    "temporary_store, shape=(2, color_block_capacity), dtype=int)\n"
)

MARKER = "color_block_capacity"


def locate() -> Path:
    try:
        import newton  # noqa: PLC0415
    except ImportError:
        sys.exit("newton is not importable. Run this with ~/IsaacLab/.venv/bin/python")
    root = Path(newton.__file__).resolve().parent
    target = root / "_src" / "solvers" / "implicit_mpm" / "solver_implicit_mpm.py"
    if not target.exists():
        sys.exit(f"not found: {target}")
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revert", action="store_true", help="restore the .bak and exit")
    a = ap.parse_args()

    target = locate()
    backup = target.with_suffix(target.suffix + ".bak")
    print(f"target: {target}")

    if a.revert:
        if not backup.exists():
            sys.exit(f"no backup to restore: {backup}")
        shutil.copy2(backup, target)
        print(f"restored from {backup}")
        return 0

    src = target.read_text()
    if MARKER in src:
        print("already patched; nothing to do")
        return 0
    if ANCHOR not in src:
        sys.exit(
            "the expected allocation was not found, so this install differs from\n"
            "1.6.1 in this region. Do NOT force it -- look at allocate_temporaries\n"
            "in the file above and port the two lines by hand:\n"
            "  partition_cell_count = "
            "self._strain_space_restriction.space_partition.geo_partition.cell_count()\n"
            "  color_block_capacity = max(strain_node_count, partition_cell_count)\n"
            "and size color_indices from color_block_capacity."
        )

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"backup: {backup}")
    target.write_text(src.replace(ANCHOR, PATCHED, 1))

    out = target.read_text()
    if MARKER not in out:
        sys.exit("write-back verification failed")
    import ast

    ast.parse(out)
    print("patched and parses clean")
    print("  color_indices now sized max(strain_node_count, partition_cell_count)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
