#!/bin/bash
# Run a command inside the Isaac Lab container against this checkout.
#
#   workstation/dev.sh bash                       # interactive shell
#   workstation/dev.sh isaaclab -p scripts/run_task.py --task Luna-Excavator-Navigate
#   workstation/dev.sh isaaclab train --rl_library rsl_rl \
#       --task Luna-Excavator-Navigate --num_envs 1024 --max_iterations 2000
#
# `isaaclab` as the first word expands to the launcher inside the image, which
# is not on PATH there.
#
# The checkout is bind-mounted, so an edit on the host is live inside the
# container: no rebuild, no reinstall, no copy. That is the entire reason to
# work locally instead of on the cluster.
#
# Kit's caches and the editable install live in $LUNA_STATE on the host, so
# they survive the container exiting. Without that, every start pays the
# five-minute cold shader compile and the package has to be reinstalled.

set -eu

IMAGE="${IMAGE:-nvcr.io/nvidia/isaac-lab:3.0.0-rc1}"
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

DISPLAY_ARGS=()
if [ -n "${DISPLAY:-}" ] && [ -d /tmp/.X11-unix ]; then
  # xhost +local:docker on the host is what actually lets the container draw.
  DISPLAY_ARGS=(-e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi

TTY_ARGS=()
[ -t 0 ] && TTY_ARGS=(-it)

exec docker run --rm "${TTY_ARGS[@]}" \
  --gpus all \
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
  "${DISPLAY_ARGS[@]}" \
  "$IMAGE" "${ARGS[@]}"
