#!/bin/bash
# Measure peak GPU memory against env count, to find out whether the MPM
# footprint scales linearly with the batch or worse than linearly.
#
#   bash workstation/memsweep.sh            # envs 1 2 4 8 16 on GPU 0
#   GPU=1 ENVS="1 2 4" bash workstation/memsweep.sh
#
# Linear growth means the simulation is simply expensive and the bed needs
# sizing to the card. Super-linear means something allocates per env that
# should be shared, which is a bug to report rather than tune around.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
ENVS="${ENVS:-1 2 4 8 16}"
TASK="${TASK:-Luna-Excavator-Excavate}"
OUT="${OUT:-$HOME/memsweep.csv}"

echo "envs,particles,peak_mib,status" > "$OUT"
for n in $ENVS; do
  log=$(mktemp); mem=$(mktemp)
  # Sample the card every 0.2 s for the whole run; the peak is what matters.
  nvidia-smi --id="$GPU" --query-gpu=memory.used \
             --format=csv,noheader,nounits -lms 200 > "$mem" 2>/dev/null &
  sampler=$!

  CUDA_VISIBLE_DEVICES="$GPU" bash "$REPO/workstation/train.sh" \
      -p "$REPO/scripts/dig_demo.py" --task "$TASK" --num_envs "$n" \
      --seconds 4 --cut 0.06 --drum -1.0 --drive 0.10 \
      > "$log" 2>&1
  rc=$?
  kill "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null

  peak=$(sort -n "$mem" | tail -1)
  parts=$(grep -o 'particles/env' -m1 "$log" >/dev/null 2>&1 \
          && grep -m2 'particles/env' "$log" | tail -1 \
             | sed 's/.*-> \([0-9]*\) particles\/env.*/\1/' || echo "?")
  if [ "$rc" -eq 0 ]; then status=ok; else status="FAILED(rc=$rc)"; fi
  printf '%s,%s,%s,%s\n' "$n" "${parts:-?}" "${peak:-?}" "$status" | tee -a "$OUT"

  # Keep the log of anything that failed; it names the allocation.
  if [ "$rc" -ne 0 ]; then cp "$log" "$HOME/memsweep_fail_${n}env.log"; fi
  rm -f "$log" "$mem"
  sleep 3   # let the driver release before the next run
done

echo
echo "wrote $OUT"
awk -F, 'NR>2 && $3 ~ /^[0-9]+$/ {
  if (prev_n && prev_m) printf "  %s -> %s envs: memory x%.2f (linear would be x%.2f)\n", prev_n, $1, $3/prev_m, $1/prev_n
} $3 ~ /^[0-9]+$/ { prev_n=$1; prev_m=$3 }' "$OUT"
