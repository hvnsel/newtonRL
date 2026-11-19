#!/bin/bash
#SBATCH -A gts-jmcnabb3
#SBATCH -q inferno
#SBATCH -J luna-play-mpm
#SBATCH -N 1 --ntasks-per-node=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o dig_mpm_play.log
#SBATCH --open-mode=append
#
# One episode of the RFT-trained policy on MPM soil, the cut set to a -0.10
# slope from 5 cm: trace, heightmap dump, GIF and video in the repo root, named
# by the MPM grid (dig_mpm_4cm.*). Needs a complete checkpoint under
# logs/rsl_rl/excavator_excavate_rft/. Submit from the repo root; every job
# appends to ./dig_mpm_play.log.
#
#   cd $REPO && sbatch cluster/play_mpm.sbatch
#   VOXEL=0.03 sbatch cluster/play_mpm.sbatch     # finer grid: contact band 1.5 cm, ~2.4x the particles
#   VIDEO=0 sbatch cluster/play_mpm.sbatch        # no video, if the node cannot render headless

set -u
VOXEL="${VOXEL:-0.04}"
VIDEO="${VIDEO:-1}"
source "$SLURM_SUBMIT_DIR/cluster/env.sh"
cd "$REPO"

NAME="dig_mpm_$(awk "BEGIN { print int($VOXEL * 100 + 0.5) }")cm"
echo "=== job $SLURM_JOB_ID on $(hostname), $(date): MPM grid $VOXEL m -> $NAME.* ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

RECORD=()
if [ "$VIDEO" = 1 ]; then
  RECORD=(--video "${NAME}_video")
fi
"$PY" -c "from isaaclab.cli import cli; cli()" -p scripts/play_dig.py \
  --task Luna-Excavator-Excavate --policy_task Luna-Excavator-Excavate-RFT --mpm_voxel "$VOXEL" \
  --no_window --slope -0.10 --depth 0.05 "${RECORD[@]}" --trace "$NAME.csv" --dump "$NAME.npz" \
  && "$PY" scripts/render_dig.py "$NAME.npz"
status=$?
if [ "$status" = 0 ] && [ "$VIDEO" = 1 ]; then
  cp "$(ls "${NAME}_video"/clip_*.mp4 | sort | tail -1)" "$NAME.mp4"
fi
echo "=== exit $status at $(date) ==="
