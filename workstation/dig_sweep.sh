#!/bin/bash
# Calibration sweep for an analytical soil tier.
#
#   bash workstation/dig_sweep.sh
#   GPU=1 OUT=~/sweep bash workstation/dig_sweep.sh
#
# Two crosses rather than a full grid, because the two cliffs we need are
# independent: resistance against cut depth, and cutting against plowing as
# the advance speed approaches the vane tip speed.
#
#   A: depth sweep at a safe advance   -> where the draft exceeds traction
#   B: advance sweep at a safe depth   -> where the tip can no longer clear
#
# Each run's stdout is kept, so a failure mid-sweep costs one point and
# scripts/parse_sweep.py can be re-run over whatever finished.

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
OUT="${OUT:-$HOME/dig_sweep}"
TASK="${TASK:-Luna-Excavator-Excavate}"
DRUM="${DRUM:--1.0}"            # full speed in the dig direction, tip 0.536 m/s
BED_AHEAD=3.19                  # m of bed in front of the drum at the dig angle
T_DRIVE=7.0

mkdir -p "$OUT"

run() {
  local cut="$1" drive="$2" tag="$3"
  # Stop once the drum has cleared the bed; driving past it measures nothing.
  local secs
  secs=$(python3 -c "
adv = $drive * 1.5
t = $T_DRIVE + min(18.0, $BED_AHEAD/adv + 2.0) if adv > 0 else 25.0
print(f'{min(t, 25.0):.1f}')")
  local log="$OUT/${tag}_cut${cut}_drive${drive}.log"
  if [ -s "$log" ] && grep -q "=== result ===" "$log"; then
    echo "  skip (done): $(basename "$log")"; return
  fi
  printf '  cut %-5s drive %-5s seconds %-5s ' "$cut" "$drive" "$secs"
  CUDA_VISIBLE_DEVICES="$GPU" bash "$REPO/workstation/train.sh" \
      -p "$REPO/scripts/dig_demo.py" --task "$TASK" --num_envs 1 \
      --cut "$cut" --drum "$DRUM" --drive "$drive" --seconds "$secs" \
      env.mpm_separate_worlds=true env.mpm_warmstart_mode=auto \
      > "$log" 2>&1
  if grep -q "=== result ===" "$log"; then echo "ok"; else echo "FAILED"; fi
  sleep 2
}

echo "A: depth sweep at drive 0.10 (advance 0.15 m/s, tip/advance 3.6)"
for cut in 0.04 0.06 0.08 0.10 0.12 0.15 0.18; do
  run "$cut" 0.10 A
done

echo "B: advance sweep at cut 0.06"
for drive in 0.05 0.15 0.25 0.35; do
  run 0.06 "$drive" B
done

echo
echo "logs in $OUT"
echo "now: python3 scripts/parse_sweep.py $OUT"
