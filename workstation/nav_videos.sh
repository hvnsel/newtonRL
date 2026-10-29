#!/bin/bash
# Navigate playback footage: N clips of one episode each, a per-step trace of
# every clip, and the graphs drawn from them.
#
#   bash workstation/nav_videos.sh
#   N=12 GPU=0 OUT=~/nav_media bash workstation/nav_videos.sh
#   CKPT=logs/rsl_rl/excavator_navigate/<run>/model_1999.pt bash workstation/nav_videos.sh
#
# Each clip is a fresh single cell of hazard terrain at full difficulty -- six
# craters and twelve boulders between 6.5 and 14 m from the centre -- with the
# goal 8-12 m from the spawn, so the drive has to cross the hazard ring. The
# camera (camera_eye in navigate_env_cfg.py) frames the whole cell from above,
# and the goal is drawn: a green disc where the machine should stop and a bar
# along the heading it should face.
#
# One clip is one episode. goal_hold_steps outlasts the episode, so after
# arriving the machine stands parked on the goal, commands latched to zero,
# until the clip ends; the clip ends 5 steps short of the 750-step timeout so
# the reset to the next spawn never shows.
#
# A clip already in OUT is skipped, so an interrupted run picks up where it
# stopped. Into OUT:
#
#   nav_01.mp4 ...   the clips, 25 fps, 1920 x 1080
#   nav_01.csv ...   per-step pose, goal, errors and speed (scripts/read_trace.py)
#   nav_01.png ...   each clip's path and errors against time
#   tracking.png arrival.png training.png summary.csv   (scripts/plot_nav.py)
#   nav_01.log ...   each run's console output

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
PY="$ISAACLAB/.venv/bin/python"
GPU="${GPU:-0}"
N="${N:-12}"
SEED0="${SEED0:-0}"             # clip i plays seed SEED0 + i
OUT="${OUT:-$HOME/nav_media}"
DMIN="${DMIN:-8.0}"
DMAX="${DMAX:-12.0}"
STEPS=745

# Newest navigate run by its timestamped name, and its last checkpoint.
if [ -z "${CKPT:-}" ]; then
  RUN="$(ls -d "$REPO"/logs/rsl_rl/excavator_navigate/20*/ 2>/dev/null | sort | tail -1)"
  CKPT="$(ls "${RUN:-.}"/model_*.pt 2>/dev/null | sort -V | tail -1)"
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "no checkpoint found under $REPO/logs/rsl_rl/excavator_navigate; set CKPT=<model_N.pt>" >&2
  exit 1
fi
# Absolute, since train.sh runs from the repo root.
CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
RUN="$(dirname "$CKPT")"
VID="$RUN/videos/play"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

echo "checkpoint $CKPT"
echo "output     $OUT"
echo "clips      $N, goals ${DMIN}-${DMAX} m, $STEPS steps each"

for i in $(seq 1 "$N"); do
  tag="$(printf 'nav_%02d' "$i")"
  seed=$((SEED0 + i))
  if [ -s "$OUT/$tag.mp4" ]; then
    echo "  $tag  skip (done)"
    continue
  fi
  stamp="$OUT/.$tag.start"
  touch "$stamp"
  printf '  %s  seed %-4s ' "$tag" "$seed"
  t0=$SECONDS
  CUDA_VISIBLE_DEVICES="$GPU" bash "$REPO/workstation/train.sh" play \
      --rl_library rsl_rl --task Luna-Excavator-Navigate --num_envs 1 \
      --checkpoint "$CKPT" --seed "$seed" --video --video_length "$STEPS" \
      env.goal_hold_steps=1000 \
      env.goal_trace_envs=1 "env.goal_trace_path=$OUT/$tag.csv" \
      "env.goal_dist_range=[$DMIN,$DMAX]" \
      env.scene.terrain.terrain_generator.num_rows=1 \
      env.scene.terrain.terrain_generator.num_cols=1 \
      env.scene.terrain.terrain_generator.use_cache=false \
      "env.scene.terrain.terrain_generator.difficulty_range=[1.0,1.0]" \
      env.scene.terrain.terrain_generator.sub_terrains.worked_light.proportion=0.0 \
      env.scene.terrain.terrain_generator.sub_terrains.worked.proportion=0.0 \
      env.scene.terrain.terrain_generator.sub_terrains.worked_heavy.proportion=0.0 \
      env.scene.terrain.max_init_terrain_level=0 \
      > "$OUT/$tag.log" 2>&1
  # The recorder numbers its clips; the one this run wrote is the one newer
  # than the stamp.
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
for clip in "$OUT"/nav_*.mp4; do
  [ -s "${clip%.mp4}.csv" ] && traces+=("${clip%.mp4}.csv")
done
if [ "${#traces[@]}" -eq 0 ]; then
  echo "no clips recorded; the logs in $OUT say why" >&2
  exit 1
fi
echo
"$PY" "$REPO/scripts/plot_nav.py" "${traces[@]}" --out "$OUT" --run "$RUN"
