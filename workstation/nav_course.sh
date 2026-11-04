#!/bin/bash
# Plays the navigate checkpoint on each mdp/course.py layout: a clip, a trace
# and the plots per layout. CKPT defaults to the newest navigate run's last
# checkpoint; clips already in OUT are skipped.
#
#   bash workstation/nav_course.sh
#   LAYOUTS="wall gate" GPU=0 OUT=~/nav_course bash workstation/nav_course.sh
#   CKPT=logs/rsl_rl/excavator_navigate/<run>/model_1999.pt bash workstation/nav_course.sh
#
# Each layout spawns the machine facing a goal 11 m ahead with obstacles on
# the line between:
#
#   boulder   one boulder on the line
#   crater    one crater on the line
#   wall      three boulders abreast
#   chicane   a boulder, then a crater, both on the line
#   gate      a row of boulders across the route with one 2 m gap off to the left
#   field     a crater and two boulders scattered over the line
#
# Into OUT:
#
#   course_<layout>.mp4   the clip, 25 fps, 1920 x 1080
#   course_<layout>.csv   per-step pose, goal, errors and speed
#   course_<layout>.png   the path over the layout, and errors against time
#   tracking.png arrival.png summary.csv   (scripts/plot_nav.py)
#   course_<layout>.log   the run's console output

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
PY="$ISAACLAB/.venv/bin/python"
GPU="${GPU:-0}"
OUT="${OUT:-$HOME/nav_course}"
LAYOUTS="${LAYOUTS:-boulder crater wall chicane gate field}"
STEPS=745

if [ -z "${CKPT:-}" ]; then
  RUN="$(ls -d "$REPO"/logs/rsl_rl/excavator_navigate/20*/ 2>/dev/null | sort | tail -1)"
  CKPT="$(ls "${RUN:-.}"/model_*.pt 2>/dev/null | sort -V | tail -1)"
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "no checkpoint found under $REPO/logs/rsl_rl/excavator_navigate; set CKPT=<model_N.pt>" >&2
  exit 1
fi
CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
VID="$(dirname "$CKPT")/videos/play"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

echo "checkpoint $CKPT"
echo "output     $OUT"
echo "layouts    $LAYOUTS"

for layout in $LAYOUTS; do
  tag="course_$layout"
  if [ -s "$OUT/$tag.mp4" ]; then
    echo "  $tag  skip (done)"
    continue
  fi
  stamp="$OUT/.$tag.start"
  touch "$stamp"
  printf '  %-15s ' "$tag"
  t0=$SECONDS
  CUDA_VISIBLE_DEVICES="$GPU" bash "$REPO/workstation/train.sh" play \
      --rl_library rsl_rl --task Luna-Excavator-Navigate-Course --num_envs 1 \
      --checkpoint "$CKPT" --seed 1 --video --video_length "$STEPS" \
      "env.scene.terrain.terrain_generator.sub_terrains.course.layout=$layout" \
      env.goal_trace_envs=1 "env.goal_trace_path=$OUT/$tag.csv" \
      > "$OUT/$tag.log" 2>&1
  # The clip this run wrote: the one newer than the stamp.
  clip="$(find "$VID" -maxdepth 1 -name '*.mp4' -newer "$stamp" 2>/dev/null | sort | tail -1)"
  rm -f "$stamp"
  if [ -n "$clip" ]; then
    mv "$clip" "$OUT/$tag.mp4"
    echo "ok   $((SECONDS - t0)) s"
  else
    echo "FAILED, see $OUT/$tag.log"
  fi
done

traces=()
for clip in "$OUT"/course_*.mp4; do
  [ -s "${clip%.mp4}.csv" ] && traces+=("${clip%.mp4}.csv")
done
if [ "${#traces[@]}" -eq 0 ]; then
  echo "no clips recorded; the logs in $OUT say why" >&2
  exit 1
fi
echo
"$PY" "$REPO/scripts/plot_nav.py" "${traces[@]}" --out "$OUT" --no-training
