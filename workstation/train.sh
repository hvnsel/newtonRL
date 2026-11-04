#!/bin/bash
# Runs the Isaac Lab CLI from $ISAACLAB/.venv (default ~/IsaacLab) in the repo
# root, so runs log to ./logs/rsl_rl/<experiment>/.
#
#   workstation/train.sh train --rl_library rsl_rl \
#       --task Luna-Excavator-Navigate --num_envs 1024 --max_iterations 400
#   workstation/train.sh play --rl_library rsl_rl --task Luna-Excavator-Navigate \
#       --num_envs 16 --checkpoint <model_N.pt>

set -eu

ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
PY="$ISAACLAB/.venv/bin/python"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO"
exec "$PY" -c "from isaaclab.cli import cli; cli()" "$@"
