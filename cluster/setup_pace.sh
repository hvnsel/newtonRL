#!/bin/bash
# One-time setup on a PACE login node (compute nodes may have no internet):
# uv and Python 3.12, Isaac Lab release/3.0.0 (kit-less, no Isaac Sim), this
# repo installed into its venv, and Newton 1.6.1 with four soil-solver files
# from 1.7.0 and one fix (scripts/patch_newton_mpm.py). Safe to re-run.
#
#   bash <repo>/cluster/setup_pace.sh

set -eu
source "$(dirname "$0")/env.sh"

# The Isaac Lab commit (release/3.0.0) this repo is developed on.
ISAACLAB_COMMIT=961760bcb91e3af48e7d116292e57e8ecaefc31e

if [ ! -x "$LUNA/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$LUNA/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv python install 3.12

if [ ! -d "$ISAACLAB/.git" ]; then
  git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB"
fi
git -C "$ISAACLAB" checkout --quiet "$ISAACLAB_COMMIT"
# PACE's glibc is 2.34; these two ship wheels for 2.35 and up only. ovstage
# serves the RTX renderer, which training never starts. omniverseclient's
# omni.client is asked about every file a USD references that is not on local
# disk, which spawning the ground plane does (its OmniPBR.mdl material);
# cluster/stubs/omni/client.py stands in for it, on PYTHONPATH from env.sh.
(cd "$ISAACLAB" && uv sync --no-install-package omniverseclient --no-install-package ovstage)

# torch's CUDA 12.8 build in place of the lock's CUDA 13.0 one: 13.0 needs a
# GPU driver of 580 or newer, 12.8 runs on PACE's drivers either side of it.
uv pip install --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"

uv pip install --python "$PY" -e "$REPO"

# Newton's soil solver for excavate: four files from 1.7.0
# (patches/newton_1.7.0/) and the colour-buffer fix, checked file by file.
# Stops the setup if the result does not match.
"$PY" "$REPO/scripts/patch_newton_mpm.py"

"$PY" -c "import torch, warp, newton, isaaclab; print('torch', torch.__version__, '| warp', warp.__version__, '| newton', newton.__version__)"
echo "setup done; next, copy in the excavator asset (cluster/ClusterSetup.md, step 3)"
