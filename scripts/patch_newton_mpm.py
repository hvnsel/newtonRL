#!/usr/bin/env python3
"""Set up the installed Newton's soil solver for the excavate task.

Isaac Lab 3.0 installs Newton 1.6.1. The excavate task runs on 1.6.1 with
four files of the implicit-MPM solver taken from Newton 1.7.0, and 1.7.0's
colour-buffer capacity fix ported into a fifth, solver_implicit_mpm.py
(1.7.0's own copy of that file needs a base class 1.6.1 lacks). Without
them the excavate env fails when the solver is built, or at its first
solve. The four files are in patches/newton_1.7.0/, unchanged from the
newton 1.7.0 wheel.

This copies them in, applies the fix, then checks every file of
newton/_src/solvers/implicit_mpm/ against its expected MD5. A file that
already matches is left alone, so a second run changes nothing. A stock
1.6.1 file is kept as <file>.bak before it is replaced. Any other Newton is
refused, with nothing changed.

  ~/IsaacLab/.venv/bin/python scripts/patch_newton_mpm.py            # install, then check
  ~/IsaacLab/.venv/bin/python scripts/patch_newton_mpm.py --check    # check only
  ~/IsaacLab/.venv/bin/python scripts/patch_newton_mpm.py --revert   # put the .bak files back

A uv sync in the Isaac Lab directory can reinstall Newton; run this again
after it. patches/newton-implicit-mpm-creep.patch is not part of this and is
not applied.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

VENDORED = Path(__file__).resolve().parents[1] / "patches" / "newton_1.7.0"

# MD5 of every file of newton/_src/solvers/implicit_mpm/ once set up.
WANT = {
    "__init__.py": "dce336d9da703f1be4eeb808a3d2a822",
    "contact_solver_kernels.py": "37bcee7d722b875ff6ad0ad2ce5c66aa",       # from 1.7.0
    "implicit_mpm_model.py": "db51b44f9cf336f0e119ee693d63e992",
    "implicit_mpm_solver_kernels.py": "79cdafdb7ecd5d4f20eee2e7ac450e8a",
    "particle_surface_colliders.py": "78371a1a3b59478c67657942c646bae5",
    "rasterized_collisions.py": "bb65dfa872aff189d78dc2e176ce6c74",        # from 1.7.0
    "render_grains.py": "e813d8ec9da965b332270ca6cc663a3a",
    "rheology_solver_kernels.py": "e232119f9d2f557dbb0473dfae323a49",      # from 1.7.0
    "solve_rheology.py": "c46a93edd20eaad601a530f480a05d3e",               # from 1.7.0
    "solver_implicit_mpm.py": "ab5d93fb5dfa077fec2959911283ff92",          # 1.6.1 with the fix
}

# Newton 1.6.1's copies of the five files this changes; the other five are
# the same in 1.6.1 and 1.7.0.
STOCK_161 = {
    "contact_solver_kernels.py": "a38b5a979a6c1b20bf956c3e3fd62823",
    "rasterized_collisions.py": "902039d11bece21b4c8573b5b12fc55a",
    "rheology_solver_kernels.py": "c9a09b81f185060caeec9c89efa00926",
    "solve_rheology.py": "660f105534640d804b23bf43de6b1c95",
    "solver_implicit_mpm.py": "c70dd8b8376d6c2aa4ab6d34f304ae83",
}

COPIED = ("contact_solver_kernels.py", "rasterized_collisions.py", "rheology_solver_kernels.py", "solve_rheology.py")

# 1.7.0's colour-buffer fix: SolverImplicitMPM.allocate_temporaries sizes
# color_indices from strain_node_count, which for a particle strain basis
# (pic8) counts particles; cell-based colouring sorts one entry per
# partition cell, and empty cells push that count higher.
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


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def locate() -> Path:
    try:
        import newton  # noqa: PLC0415
    except ImportError:
        sys.exit("newton is not importable. Run this with the Isaac Lab venv's python, "
                 "e.g. ~/IsaacLab/.venv/bin/python")
    folder = Path(newton.__file__).resolve().parent / "_src" / "solvers" / "implicit_mpm"
    if not folder.is_dir():
        sys.exit(f"not found: {folder}")
    return folder


def check(folder: Path) -> int:
    """Print each file against its expected MD5; 0 if all match."""
    off = []
    for name, want in WANT.items():
        path = folder / name
        ok = path.exists() and md5(path.read_bytes()) == want
        print(f"  {'ok     ' if ok else 'DIFFERS'}  {name}")
        if not ok:
            off.append(name)
    if off:
        print(f"{len(off)} of {len(WANT)} files differ: Newton is not set up for excavate")
        return 1
    print(f"all {len(WANT)} files match: Newton is set up for excavate")
    return 0


def install(folder: Path) -> int:
    have = {name: md5((folder / name).read_bytes()) if (folder / name).exists() else "missing"
            for name in WANT}
    unknown = [name for name, h in have.items() if h not in (WANT[name], STOCK_161.get(name, WANT[name]))]
    if unknown:
        print("This Newton is neither Isaac Lab 3.0's 1.6.1 nor already set up; nothing changed.")
        for name in unknown:
            print(f"  {name}: md5 {have[name]}")
        print("To put back Isaac Lab's 1.6.1, then run this again:")
        print(f"  uv pip install --python {sys.executable} --reinstall --no-deps newton==1.6.1")
        return 1

    def backup(path: Path) -> None:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            shutil.copyfile(path, bak)

    changed = 0
    for name in COPIED:
        if have[name] == WANT[name]:
            continue
        src = VENDORED / name
        if not src.exists():
            sys.exit(f"missing {src}; it ships with this repo")
        # LF as in the wheel, whatever the checkout did to line endings.
        data = src.read_bytes().replace(b"\r\n", b"\n")
        if md5(data) != WANT[name]:
            sys.exit(f"{src} is not Newton 1.7.0's {name}; restore it with git checkout")
        backup(folder / name)
        (folder / name).write_bytes(data)
        print(f"copied   {name} from Newton 1.7.0")
        changed += 1

    target = folder / "solver_implicit_mpm.py"
    if have["solver_implicit_mpm.py"] != WANT["solver_implicit_mpm.py"]:
        data = target.read_bytes()
        backup(target)
        target.write_bytes(data.replace(ANCHOR.encode(), PATCHED.encode(), 1))
        print("patched  solver_implicit_mpm.py with 1.7.0's colour-buffer fix")
        changed += 1

    if not changed:
        print("already set up; nothing changed")
    return check(folder)


def revert(folder: Path) -> int:
    baks = sorted(folder.glob("*.py.bak"))
    if not baks:
        print("no .bak files to restore")
        return 1
    for bak in baks:
        shutil.copyfile(bak, bak.with_suffix(""))
        print(f"restored {bak.with_suffix('').name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare only, change nothing")
    ap.add_argument("--revert", action="store_true", help="restore the .bak files")
    a = ap.parse_args()

    folder = locate()
    print(f"Newton soil solver: {folder}")
    if a.revert:
        return revert(folder)
    if a.check:
        return check(folder)
    return install(folder)


if __name__ == "__main__":
    raise SystemExit(main())
