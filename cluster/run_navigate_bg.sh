#!/bin/bash
# Launches a Luna-Excavator-Navigate training run detached (nohup), inside an
# allocation you already hold. Prints the PID, log path, GPU, commit and key
# config values, then returns.
#
#   bash cluster/run_navigate_bg.sh runE
#   ENVS=2048 ITERS=4000 bash cluster/run_navigate_bg.sh runF
#
# Killing the training process (or suspending it with `kill -STOP <pid>`)
# keeps the allocation; start the next run with this script.
#
# Sets its own environment and the same binds as train_navigate.sbatch.

set -u
TAG="${1:-run}"
ENVS="${ENVS:-1024}"
ITERS="${ITERS:-2000}"

export LUNA_SCRATCH=/storage/scratch1/5/$USER/luna
export REPO=$LUNA_SCRATCH/Luna_HiFi
export PYTHONUSERBASE=$LUNA_SCRATCH/pyuser
export XDG_CACHE_HOME=$LUNA_SCRATCH/cache/xdg
export TMPDIR=$LUNA_SCRATCH/tmp
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y PRIVACY_CONSENT=Y

OV=$LUNA_SCRATCH/cache/ov-container
mkdir -p "$OV"/{kit-cache,kit-logs,kit-data,nv-omniverse,ov-cache} \
         "$PYTHONUSERBASE" "$TMPDIR"
BINDS="-B $LUNA_SCRATCH:$LUNA_SCRATCH"
BINDS="$BINDS -B $OV/kit-cache:/isaac-sim/kit/cache"
BINDS="$BINDS -B $OV/kit-logs:/isaac-sim/kit/logs"
BINDS="$BINDS -B $OV/kit-data:/isaac-sim/kit/data"
BINDS="$BINDS -B $OV/nv-omniverse:/root/.nvidia-omniverse"
BINDS="$BINDS -B $OV/ov-cache:/root/.cache/ov"

# The image runs from node-local disk, copied once per node.
SIF=/tmp/isaaclab.sif
[ -f "$SIF" ] || cp "$LUNA_SCRATCH/isaaclab.sif" "$SIF"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)

cd "$REPO"

LOG=$LUNA_SCRATCH/$TAG-$(date +%m%d-%H%M).log
nohup apptainer exec --nv $BINDS "$SIF" \
  /workspace/isaaclab/isaaclab.sh train --rl_library rsl_rl \
  --task Luna-Excavator-Navigate \
  --num_envs "$ENVS" \
  --max_iterations "$ITERS" \
  > "$LOG" 2>&1 &

echo "pid    $!"
echo "log    $LOG"
echo "gpu    $GPU"
echo "envs   $ENVS   iters $ITERS"
# " +local" marks uncommitted changes under luna_hifi_tasks/.
DIRTY=$(git diff --quiet HEAD -- luna_hifi_tasks || echo " +local")
echo "head   $(git rev-parse --short HEAD)$DIRTY"
echo "config $(grep -hoE '(schedule|learning_rate|entropy_coef)=[^,]+' \
  luna_hifi_tasks/excavator/navigate/agents/rsl_rl_ppo_cfg.py | tr '\n' ' ')"
echo "       $(grep -hoE '(w_goal|goal_heading_tol_rad) = \S+' \
  luna_hifi_tasks/excavator/navigate/navigate_env_cfg.py | tr '\n' ' ')"
