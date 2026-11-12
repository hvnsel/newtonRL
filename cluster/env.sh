# Paths and caches for this repo on PACE, all in scratch ($HOME holds 20 GB).
# Sourced by setup_pace.sh and the sbatch scripts; source it in any shell that
# runs this repo by hand.

export LUNA="${LUNA:-/storage/scratch1/5/$USER/luna}"
export REPO="$LUNA/newtonRL"
export ISAACLAB="$LUNA/IsaacLab"
export PY="$ISAACLAB/.venv/bin/python"
export PATH="$LUNA/bin:$PATH"
# A stand-in for omni.client, whose wheel PACE cannot install (stubs/omni/client.py).
export PYTHONPATH="$REPO/cluster/stubs${PYTHONPATH:+:$PYTHONPATH}"

export UV_CACHE_DIR="$LUNA/cache/uv"
export UV_PYTHON_INSTALL_DIR="$LUNA/python"
export XDG_CACHE_HOME="$LUNA/cache/xdg"
export WARP_CACHE_PATH="$LUNA/cache/warp"
export TMPDIR="$LUNA/tmp"
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y

mkdir -p "$LUNA/bin" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" "$WARP_CACHE_PATH" "$TMPDIR"
