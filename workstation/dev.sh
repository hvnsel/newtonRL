#!/bin/bash
# Runs a command in the Isaac Lab container with this checkout bind-mounted at
# /workspace/luna. Kit caches and the user site-packages persist in
# $LUNA_STATE (default ~/.luna).
#
#   workstation/dev.sh bash                       # interactive shell
#   workstation/dev.sh isaaclab train --rl_library rsl_rl \
#       --task Luna-Excavator-Navigate --num_envs 1024 --max_iterations 2000
#
# A leading `isaaclab` runs the image's isaaclab.sh. CONTAINER=podman uses
# podman instead of docker.

set -eu

IMAGE="${IMAGE:-nvcr.io/nvidia/isaac-lab:3.0.0-rc1}"
CONTAINER="${CONTAINER:-docker}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE="${LUNA_STATE:-$HOME/.luna}"
MOUNT=/workspace/luna

mkdir -p "$STATE"/{kit-cache,kit-logs,kit-data,nv-omniverse,ov-cache,pyuser,tmp}

ARGS=()
if [ "${1:-}" = "isaaclab" ]; then
  shift
  ARGS=(/workspace/isaaclab/isaaclab.sh "$@")
elif [ "$#" -eq 0 ]; then
  ARGS=(bash)
else
  ARGS=("$@")
fi

TTY_ARGS=()
[ -t 0 ] && TTY_ARGS=(-it)

# podman takes GPUs as CDI devices.
GPU_ARGS=(--gpus all)
[ "$CONTAINER" = podman ] && GPU_ARGS=(--device nvidia.com/gpu=all)

exec "$CONTAINER" run --rm "${TTY_ARGS[@]}" \
  "${GPU_ARGS[@]}" \
  --ipc host \
  -v "$REPO:$MOUNT" \
  -w "$MOUNT" \
  -v "$STATE/kit-cache:/isaac-sim/kit/cache" \
  -v "$STATE/kit-logs:/isaac-sim/kit/logs" \
  -v "$STATE/kit-data:/isaac-sim/kit/data" \
  -v "$STATE/nv-omniverse:/root/.nvidia-omniverse" \
  -v "$STATE/ov-cache:/root/.cache/ov" \
  -v "$STATE/pyuser:/root/.local" \
  -v "$STATE/tmp:/tmp/luna" \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e TMPDIR=/tmp/luna \
  "$IMAGE" "${ARGS[@]}"
