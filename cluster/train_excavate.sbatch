#!/bin/bash
#SBATCH -A gts-jmcnabb3
#SBATCH -q inferno
#SBATCH -J luna-excavate
#SBATCH -N 1 --ntasks-per-node=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o dig_train.log
#SBATCH --open-mode=append
#
# Trains Luna-Excavator-Excavate up to TOTAL iterations. It resumes from the
# newest checkpoint that loads, so a job cut off at 12 h continues in the next.
# Submit from the repo root; every job appends to ./dig_train.log.
#
#   cd $REPO && sbatch cluster/train_excavate.sbatch
#   TOTAL=300 sbatch cluster/train_excavate.sbatch
#   FRESH=1 sbatch cluster/train_excavate.sbatch      # new run, ignoring checkpoints
#
# Three jobs back to back, each starting when the one before ends:
#   j=$(sbatch --parsable cluster/train_excavate.sbatch); for i in 2 3; do j=$(sbatch --parsable --dependency=afterany:$j cluster/train_excavate.sbatch); done

set -u
TOTAL="${TOTAL:-200}"
ENVS="${ENVS:-4}"
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$REPO"

echo "=== job $SLURM_JOB_ID on $(hostname), $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

N=0
RESUME=()
if [ "${FRESH:-0}" != 1 ]; then
  for f in $(ls -t logs/rsl_rl/excavator_excavate/*/model_*.pt 2>/dev/null); do
    if "$PY" -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" "$f" 2>/dev/null; then
      N=$(basename "$f" .pt | cut -d_ -f2)
      RESUME=(--checkpoint "$f")
      break
    fi
    echo "skip $f (cut short)"
  done
fi

LEFT=$((TOTAL - N))
if [ "$LEFT" -le 1 ]; then
  echo "already at $N of $TOTAL iterations; nothing to do"
  exit 0
fi
echo "start at iteration $N, $LEFT to go, from ${RESUME[1]:-scratch}"

"$PY" -c "from isaaclab.cli import cli; cli()" train --rl_library rsl_rl \
  --task Luna-Excavator-Excavate --num_envs "$ENVS" --max_iterations "$LEFT" "${RESUME[@]}"
echo "=== exit $? at $(date) ==="
