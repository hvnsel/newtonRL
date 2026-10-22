#!/bin/bash
# Run the Isaac Lab CLI against this checkout, on a workstation install.
#
#   workstation/train.sh train --rl_library rsl_rl \
#       --task Luna-Excavator-Navigate --num_envs 1024 --max_iterations 400 \
#       env.actor_scan=false
#
#   workstation/train.sh -p scripts/run_task.py --task Luna-Excavator-Navigate --num_envs 4
#
# Why this exists rather than isaaclab.sh: that wrapper resolves its own
# interpreter and on a box whose system python is older than the venv's it
# picks the wrong one, failing with "No module named 'tomllib'" -- stdlib from
# 3.11 on, so the message means 3.10 ran it. It is also deprecated in favour of
# `uv run isaaclab`, which resolves the project from the working directory and
# so wants to be run from the Isaac Lab tree, not from here.
#
# Staying in the repo root is the point: the trainer writes checkpoints to
# ./logs/rsl_rl/<experiment>/, which is where scripts/peek.py looks.

set -eu

ISAACLAB="${ISAACLAB:-$HOME/IsaacLab}"
PY="$ISAACLAB/.venv/bin/python"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -x "$PY" ]; then
  echo "no interpreter at $PY" >&2
  echo "Set ISAACLAB to the Isaac Lab checkout, or see workstation/Setup.md." >&2
  exit 1
fi

# The venv has to be new enough to run the CLI at all; the failure otherwise is
# a missing-stdlib-module error that says nothing about versions.
"$PY" - <<'PYCHECK' || exit 1
import sys
if sys.version_info < (3, 12):
    sys.exit(
        f"the venv is Python {sys.version_info.major}.{sys.version_info.minor}; "
        "Isaac Lab 3.0 needs >=3.12,<3.13"
    )
PYCHECK

cd "$REPO"
exec "$PY" -c "from isaaclab.cli import cli; cli()" "$@"
