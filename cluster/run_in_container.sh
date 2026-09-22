#!/usr/bin/env bash
# cluster/run_in_container.sh -- run something inside the Isaac Lab image.
#
#   source cluster/pace_env.sh
#   bash cluster/run_in_container.sh newton_smoke.py
#   bash cluster/run_in_container.sh /workspace/isaaclab/scripts/.../train.py --task ...
#
# Exists because three details are easy to get wrong and each one fails in a
# way that does not name itself:
#
#   --nv          without it a GPU node reports no CUDA device and Kit dies on
#                 "Found no NVIDIA driver", which reads as a broken image.
#   the python    there is no `python` on PATH inside the image; Isaac Sim
#                 bundles its own interpreter.
#   the binds     Kit's caches live at fixed paths INSIDE the read-only
#                 squashfs. Unbound, every launch recompiles its shaders and
#                 logs "Failed to acquire exclusive lock to data store". It
#                 still runs -- it just pays several minutes, every time.
set -euo pipefail

: "${LUNA_SCRATCH:?source cluster/pace_env.sh first}"
SIF="${LUNA_SIF:-$LUNA_SCRATCH/isaaclab.sif}"
[ -f "$SIF" ] || { echo "no image at $SIF (set LUNA_SIF)" >&2; exit 1; }

# One writable tree that shadows every fixed path Kit insists on owning.
OV="$LUNA_SCRATCH/cache/ov-container"
mkdir -p "$OV"/{kit-cache,kit-logs,kit-data,nv-omniverse,ov-cache,ov-data,ov-logs}

# -B src:dst. Each dst is a directory in the image that Kit writes to and
# cannot, because a squashfs is read-only.
BINDS=(
  -B "$LUNA_SCRATCH:$LUNA_SCRATCH"
  -B "$OV/kit-cache:/isaac-sim/kit/cache"
  -B "$OV/kit-logs:/isaac-sim/kit/logs"
  -B "$OV/kit-data:/isaac-sim/kit/data"
  -B "$OV/nv-omniverse:/root/.nvidia-omniverse"
  -B "$OV/ov-cache:/root/.cache/ov"
  -B "$OV/ov-local:/root/.local/share/ov"
)
mkdir -p "$OV/ov-local"

# Isaac Sim ignores XDG for some of its own caches but Warp does respect it,
# and Warp's kernel cache is the one that quietly eats a 20 GB home quota.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$LUNA_SCRATCH/cache/xdg}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$LUNA_SCRATCH/cache/warp}"
mkdir -p "$XDG_CACHE_HOME" "$WARP_CACHE_PATH"

echo "[run] image   $SIF"
echo "[run] caches  $OV"
echo "[run] first launch compiles shaders and takes minutes; later ones do not"

exec apptainer exec --nv "${BINDS[@]}" "$SIF" \
  /workspace/isaaclab/isaaclab.sh -p "$@"
