#!/bin/bash
#SBATCH -A gts-jmcnabb3
#SBATCH -q inferno
#SBATCH -J luna-navigate
#SBATCH -N 1 --ntasks-per-node=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o train.log
#
# Trains Luna-Excavator-Navigate from scratch. Submit from the repo root;
# output goes to ./train.log.
#
#   cd $REPO && sbatch cluster/train_navigate.sbatch
#   ENVS=1024 ITERS=4000 sbatch cluster/train_navigate.sbatch
#   EXTRA="env.w_goal=50" sbatch cluster/train_navigate.sbatch     # Hydra overrides

set -u
ENVS="${ENVS:-1024}"
ITERS="${ITERS:-4000}"
EXTRA="${EXTRA:-}"
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$REPO"

echo "=== job $SLURM_JOB_ID on $(hostname), $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "envs $ENVS, iterations $ITERS, overrides: ${EXTRA:-none}"

"$PY" -c "from isaaclab.cli import cli; cli()" train --rl_library rsl_rl \
  --task Luna-Excavator-Navigate --num_envs "$ENVS" --max_iterations "$ITERS" $EXTRA
echo "=== exit $? at $(date) ==="
