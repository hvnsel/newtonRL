#!/usr/bin/env bash
# cluster/pace_probe.sh -- print the facts needed to write a correct job script.
#
#   bash cluster/pace_probe.sh
#
# Run this on a LOGIN node and paste the output. Everything it asks about is
# site-specific -- partition names, which GPUs exist and how they are
# requested, whether Apptainer is a module or already on PATH -- and all of it
# is cheap to look up and expensive to assume.

echo "=== who and where ==="
echo "user      : $USER"
echo "host      : $(hostname)"
echo "home quota: $(df -h "$HOME" 2>/dev/null | tail -1)"

echo; echo "=== charge accounts (the -A flag) ==="
sacctmgr -nP show assoc user="$USER" format=Account,Partition,QOS 2>/dev/null | sort -u \
  || echo "  sacctmgr unavailable; use the account from pace-quota"

echo; echo "=== partitions ==="
sinfo -o "%20P %10a %12l %10D %25f %N" 2>/dev/null | head -30

echo; echo "=== GPU types, and how to ask for them ==="
# GRES strings are what --gres=gpu:<type>:<n> must match.
sinfo -o "%25N %10c %10m %40G" 2>/dev/null | grep -i gpu | head -20 \
  || echo "  no gres info from sinfo"

echo; echo "=== apptainer / singularity ==="
command -v apptainer  && apptainer  version
command -v singularity && singularity version
echo "-- modules matching container runtimes --"
( module -t avail 2>&1 | grep -iE "apptainer|singularity" ) || echo "  none found by name"

echo; echo "=== CUDA-capable modules (for the driver/toolkit the image needs) ==="
( module -t avail 2>&1 | grep -iE "^cuda|nvhpc" | head -10 ) || echo "  none"

echo; echo "=== scratch purge policy ==="
ls -d /storage/scratch1/*/"$USER" 2>/dev/null
echo "  (PACE purges scratch on an access-time policy -- confirm the window in"
echo "   the docs; anything you want to keep belongs in project storage.)"

echo; echo "=== what a GPU node actually has ==="
echo "  Not visible from a login node. Get it with an interactive job:"
echo "    salloc -A <account> -N1 --gres=gpu:1 -t0:20:00"
echo "    nvidia-smi; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"
echo "  The driver version is the one thing that can veto an Isaac Sim image."
