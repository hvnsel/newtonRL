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
# Trains Luna-Excavator-Navigate unattended.
#
#   sbatch cluster/train_navigate.sbatch              # defaults below
#   ENVS=2048 ITERS=4000 sbatch cluster/train_navigate.sbatch
#   squeue -u $USER
#   tail -f luna-navigate-*.out
#
# Checkpoints: every save_interval iterations (navigate/agents/rsl_rl_ppo_cfg.py)
# under $REPO/logs/rsl_rl/excavator_navigate/<timestamp>/.
#
# --gres sets the GPU type; the image needs compute capability >= 7.5.
#   sinfo -o "%.20P %.18G" | grep -i gpu      lists the GPU types.

set -u

ENVS="${ENVS:-1024}"
ITERS="${ITERS:-2000}"
# Hydra overrides, e.g. EXTRA="env.w_goal=50 agent.algorithm.entropy_coef=0.01": env
# config under env., PPO config under agent.
EXTRA="${EXTRA:-}"

export LUNA_SCRATCH=/storage/scratch1/5/$USER/luna
export REPO=$LUNA_SCRATCH/Luna_HiFi
export PYTHONUSERBASE=$LUNA_SCRATCH/pyuser
export XDG_CACHE_HOME=$LUNA_SCRATCH/cache/xdg
export TMPDIR=$LUNA_SCRATCH/tmp
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y PRIVACY_CONSENT=Y

# Binds Kit's cache paths inside the read-only image to scratch.
OV=$LUNA_SCRATCH/cache/ov-container
mkdir -p "$OV"/{kit-cache,kit-logs,kit-data,nv-omniverse,ov-cache} \
         "$PYTHONUSERBASE" "$TMPDIR"
BINDS="-B $LUNA_SCRATCH:$LUNA_SCRATCH"
BINDS="$BINDS -B $OV/kit-cache:/isaac-sim/kit/cache"
BINDS="$BINDS -B $OV/kit-logs:/isaac-sim/kit/logs"
BINDS="$BINDS -B $OV/kit-data:/isaac-sim/kit/data"
BINDS="$BINDS -B $OV/nv-omniverse:/root/.nvidia-omniverse"
BINDS="$BINDS -B $OV/ov-cache:/root/.cache/ov"

# The image runs from node-local disk.
SIF=/tmp/isaaclab.sif
cp "$LUNA_SCRATCH/isaaclab.sif" "$SIF"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "envs=$ENVS iterations=$ITERS repo=$REPO extra=${EXTRA:-none}"

cd "$REPO"
apptainer exec --nv $BINDS "$SIF" \
  /workspace/isaaclab/isaaclab.sh train --rl_library rsl_rl \
  --task Luna-Excavator-Navigate \
  --num_envs "$ENVS" \
  --max_iterations "$ITERS" \
  $EXTRA

echo "exit $?"
ls -lh logs/rsl_rl/excavator_navigate/*/ | tail -20
