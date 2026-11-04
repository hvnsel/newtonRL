#!/bin/bash
#SBATCH -A gts-jmcnabb3
#SBATCH -J isaaclab-sif
#SBATCH -N 1 -n 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out
#
# Builds the Isaac Lab .sif on node-local /tmp (CPU only) and copies it to
# scratch.
#
#   sbatch cluster/build_sif.sbatch
#   squeue -u $USER
#   tail -f isaaclab-sif-*.out

set -u
IMAGE="docker://nvcr.io/nvidia/isaac-lab:3.0.0-rc1"
DEST="/storage/scratch1/5/$USER/luna/isaaclab.sif"
LOCAL="/tmp/sifbuild-$SLURM_JOB_ID"

# Layer cache on scratch; build files on node-local disk.
export APPTAINER_CACHEDIR="/storage/scratch1/5/$USER/luna/cache/apptainer"
export APPTAINER_TMPDIR="$LOCAL/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$(dirname "$DEST")"

echo "=== node $(hostname), job $SLURM_JOB_ID, building in $LOCAL ==="
df -h /tmp "$APPTAINER_CACHEDIR"

start=$SECONDS
apptainer pull --force "$LOCAL/isaaclab.sif" "$IMAGE"
rc=$?
echo "=== pull exited $rc after $(( (SECONDS-start)/60 )) min ==="
[ $rc -eq 0 ] || exit $rc

cp "$LOCAL/isaaclab.sif" "$DEST" && rm -rf "$LOCAL"
ls -lh "$DEST"
echo "=== done. Verify with: apptainer exec $DEST /workspace/isaaclab/isaaclab.sh -p -c 'import isaaclab' ==="
