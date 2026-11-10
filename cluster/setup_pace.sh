#!/bin/bash
# One-time setup on a PACE login node (compute nodes may have no internet):
# uv and Python 3.12, Isaac Lab at the workstation's commit (kit-less, no Isaac
# Sim), this repo installed into its venv, and the workstation's Newton: 1.6.1
# with 1.7.0's rheology kernels and scripts/patch_newton_mpm.py. Safe to re-run.
#
#   bash /storage/scratch1/5/$USER/luna/newtonRL/cluster/setup_pace.sh

set -eu
source "$(dirname "$0")/env.sh"

# The workstation's ~/IsaacLab commit (release/3.0.0).
ISAACLAB_COMMIT=961760bcb91e3af48e7d116292e57e8ecaefc31e

if [ ! -x "$LUNA/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$LUNA/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv python install 3.12

if [ ! -d "$ISAACLAB/.git" ]; then
  git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB"
fi
git -C "$ISAACLAB" checkout --quiet "$ISAACLAB_COMMIT"
# PACE's glibc is 2.34; these two ship wheels for 2.35 and up only. Training
# uses neither: omniverseclient fetches remote Nucleus/S3 assets, ovstage
# serves the RTX renderer, and both are imported only on those paths.
(cd "$ISAACLAB" && uv sync --no-install-package omniverseclient --no-install-package ovstage)

uv pip install --python "$PY" -e "$REPO"

# The workstation runs Newton 1.6.1 with 1.7.0's rheology kernels, plus the
# colour-buffer fix. The file is replaced, not rewritten, so uv's cache keeps
# the original.
MPM_DIR=$("$PY" -c "import newton, os; print(os.path.join(os.path.dirname(newton.__file__), '_src', 'solvers', 'implicit_mpm'))")
curl -sSfL https://raw.githubusercontent.com/newton-physics/newton/v1.7.0/newton/_src/solvers/implicit_mpm/rheology_solver_kernels.py \
  -o "$TMPDIR/rheology_solver_kernels.py"
mv "$TMPDIR/rheology_solver_kernels.py" "$MPM_DIR/rheology_solver_kernels.py"
"$PY" "$REPO/scripts/patch_newton_mpm.py"

"$PY" -c "import torch, warp, newton, isaaclab; print('torch', torch.__version__, '| warp', warp.__version__, '| newton', newton.__version__)"
echo "setup done; next, copy the excavator asset from the workstation (cluster/ClusterSetup.md, step 3)"
