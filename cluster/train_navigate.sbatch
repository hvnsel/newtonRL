#!/bin/bash
#SBATCH -A gts-jmcnabb3
#SBATCH -q inferno
#SBATCH -J luna-navigate
#SBATCH -N 1 --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out
#
# Train Luna-Excavator-Navigate unattended.
#
#   sbatch cluster/train_navigate.sbatch              # defaults below
#   ENVS=2048 ITERS=4000 sbatch cluster/train_navigate.sbatch
#   squeue -u $USER
#   tail -f luna-navigate-*.out
#
# Checkpoints land every save_interval iterations (100, set in
# navigate/agents/rsl_rl_ppo_cfg.py) under
# $REPO/logs/rsl_rl/excavator_navigate/<timestamp>/.
#
# The GPU is pinned by TYPE, not partition: PACE treats -p as a hint and
# reassigns the job, so -p gpu-l40s lands on gpu-v100 and the run dies on
# "no kernel image is available" -- the image's PyTorch is built for compute
# capability 7.5 and up, and a V100 is 7.0.
#
# sinfo -o "%.20P %.18G" | grep -i gpu   lists the types this cluster offers.

set -u

ENVS="${ENVS:-1024}"
ITERS="${ITERS:-2000}"

export LUNA_SCRATCH=/storage/scratch1/5/$USER/luna
export REPO=$LUNA_SCRATCH/Luna_HiFi
export PYTHONUSERBASE=$LUNA_SCRATCH/pyuser
export XDG_CACHE_HOME=$LUNA_SCRATCH/cache/xdg
export TMPDIR=$LUNA_SCRATCH/tmp
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y PRIVACY_CONSENT=Y

# Kit writes caches to fixed paths inside the read-only image. These binds
# redirect them to scratch.
OV=$LUNA_SCRATCH/cache/ov-container
mkdir -p "$OV"/{kit-cache,kit-logs,kit-data,nv-omniverse,ov-cache} \
         "$PYTHONUSERBASE" "$TMPDIR"
BINDS="-B $LUNA_SCRATCH:$LUNA_SCRATCH"
BINDS="$BINDS -B $OV/kit-cache:/isaac-sim/kit/cache"
BINDS="$BINDS -B $OV/kit-logs:/isaac-sim/kit/logs"
BINDS="$BINDS -B $OV/kit-data:/isaac-sim/kit/data"
BINDS="$BINDS -B $OV/nv-omniverse:/root/.nvidia-omniverse"
BINDS="$BINDS -B $OV/ov-cache:/root/.cache/ov"

# The image is read off node-local disk. Reading 12 GB over Lustre on every
# kernel launch is what makes a cold start take five minutes instead of one.
SIF=/tmp/isaaclab.sif
cp "$LUNA_SCRATCH/isaaclab.sif" "$SIF"

# usd_status() compares mtimes, and a git checkout stamps every file at the
# same second.
touch "$REPO/assets/excavator/excavator.usda"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)
case "$GPU" in
  *V100*|*P100*|*K80*)
    echo "FATAL: $GPU is compute capability < 7.5; the image has no kernels for it."
    exit 1 ;;
esac

echo "envs=$ENVS iterations=$ITERS repo=$REPO"

cd "$REPO"
apptainer exec --nv $BINDS "$SIF" \
  /workspace/isaaclab/isaaclab.sh train --rl_library rsl_rl \
  --task Luna-Excavator-Navigate \
  --num_envs "$ENVS" \
  --max_iterations "$ITERS"

echo "exit $?"
ls -lh logs/rsl_rl/excavator_navigate/*/ | tail -20
