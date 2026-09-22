#!/bin/bash
#SBATCH -A gts-jmcnabb3
#SBATCH -J isaaclab-sif
#SBATCH -N 1 -n 8
#SBATCH --mem=48G
#SBATCH -t 8:00:00
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.out
#
# Build the Isaac Lab .sif unattended.
#
#   sbatch cluster/build_sif.sbatch
#   squeue -u $USER          # watch
#   tail -f isaaclab-sif-*.out
#
# No GPU requested on purpose: this is a download and a compression, and
# holding a GPU through it wastes credits for nothing.
#
# The reason this is a batch job rather than something you sit through:
# mksquashfs writes many small blocks, and a parallel filesystem like Lustre is
# built for large sequential I/O. Writing the image directly to scratch was
# quoting 22+ hours. Node-local disk turns that into minutes, and the finished
# .sif then moves to scratch as ONE sequential copy, which Lustre is good at.

set -u
IMAGE="docker://nvcr.io/nvidia/isaac-lab:3.0.0-rc1"
DEST="/storage/scratch1/5/$USER/luna/isaaclab.sif"
NEED_GB=70          # unpacked rootfs and the .sif have to coexist

# The layer cache stays on scratch: it is 15 GB, it is already populated, and
# keeping it means a retry never re-downloads.
export APPTAINER_CACHEDIR="/storage/scratch1/5/$USER/luna/cache/apptainer"
mkdir -p "$APPTAINER_CACHEDIR" "$(dirname "$DEST")"

# Find node-local scratch with room. Slurm usually sets TMPDIR to exactly this;
# the others are common fallbacks. Whatever is picked, it must be LOCAL -- a
# path that happens to live on the shared filesystem defeats the whole point.
pick_local() {
  for d in "${TMPDIR:-}" "/local/scratch/$USER" "/local/$USER" /local/scratch /local /tmp; do
    [ -n "$d" ] || continue
    mkdir -p "$d/sifbuild-$SLURM_JOB_ID" 2>/dev/null || continue
    avail=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc '0-9')
    [ -n "${avail:-}" ] || continue
    if [ "$avail" -ge "$NEED_GB" ]; then
      echo "$d/sifbuild-$SLURM_JOB_ID"
      return 0
    fi
    echo "  $d has only ${avail}G free, need ${NEED_GB}G" >&2
    rmdir "$d/sifbuild-$SLURM_JOB_ID" 2>/dev/null
  done
  return 1
}

echo "=== node $(hostname), job $SLURM_JOB_ID ==="
echo "looking for node-local scratch with ${NEED_GB}G free..."
if LOCAL=$(pick_local); then
  echo "building in $LOCAL"
  export APPTAINER_TMPDIR="$LOCAL/tmp"
  BUILD="$LOCAL/isaaclab.sif"
  mkdir -p "$APPTAINER_TMPDIR"
else
  echo "!! no node-local disk with ${NEED_GB}G. Falling back to scratch, which"
  echo "!! is the slow path -- 8 hours may not be enough. If this times out,"
  echo "!! rerun with --sandbox instead (see the note at the end of this file)."
  export APPTAINER_TMPDIR="/storage/scratch1/5/$USER/luna/tmp"
  BUILD="$DEST"
  mkdir -p "$APPTAINER_TMPDIR"
fi

echo "cache  $APPTAINER_CACHEDIR"
echo "tmp    $APPTAINER_TMPDIR"
echo "build  $BUILD"
df -h "$(dirname "$BUILD")" "$APPTAINER_CACHEDIR"

start=$SECONDS
apptainer pull --force "$BUILD" "$IMAGE"
rc=$?
echo "=== pull exited $rc after $(( (SECONDS-start)/60 )) min ==="
[ $rc -eq 0 ] || exit $rc

if [ "$BUILD" != "$DEST" ]; then
  echo "copying to scratch (one sequential write, which Lustre likes)..."
  cp "$BUILD" "$DEST" && rm -f "$BUILD"
  rm -rf "$APPTAINER_TMPDIR"
fi

ls -lh "$DEST"
echo "=== done. Verify with: apptainer exec $DEST python -c 'import isaaclab' ==="

# If this times out even on local disk, the escape hatch is a SANDBOX: a plain
# directory instead of a compressed image, which skips mksquashfs entirely.
#
#   apptainer build --sandbox "$LOCAL/isaaclab" "$IMAGE"
#
# It runs fine and starts faster. The cost is hundreds of thousands of small
# files, against a scratch file-count quota of 1,000,000 -- so keep it on local
# disk if you can, and check `pace-quota` afterwards if you cannot.
