#!/bin/bash
# Launch a navigate run detached, inside an allocation that already exists.
#
#   bash cluster/run_navigate_bg.sh runE
#   ENVS=2048 ITERS=4000 bash cluster/run_navigate_bg.sh runF
#
# Prints the PID, the log path and the commit it is running, then returns.
#
# This is how a run gets swapped without losing the allocation. What holds a
# Slurm job open is the sbatch script's own process, not the training process,
# so the training process can be killed -- or suspended with
# `kill -STOP <pid>`, which leaves it at STAT Tl and frees the GPU -- and
# another started here. Queue waits run one to two days, so the allocation is
# worth more than any single run inside it.
#
# Everything the container needs is set here rather than inherited, so this
# works from any shell on the node: the binds are the ones train_navigate.sbatch
# uses, and Kit writes its caches to fixed paths inside a read-only image
# without them.

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

# Node-local, so a run on a fresh node copies it once. Reading 12 GB over
# Lustre on every kernel launch is a five-minute cold start instead of one.
SIF=/tmp/isaaclab.sif
[ -f "$SIF" ] || cp "$LUNA_SCRATCH/isaaclab.sif" "$SIF"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)
case "$GPU" in
  *V100*|*P100*|*K80*)
    echo "FATAL: $GPU is compute capability < 7.5; the image has no kernels for it."
    exit 1 ;;
esac

cd "$REPO"
# usd_status() compares mtimes, and a git checkout stamps every file at the
# same second.
touch assets/excavator/excavator.usda

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
echo "commit $(git rev-parse --short HEAD)  $(git log -1 --format=%s)"
