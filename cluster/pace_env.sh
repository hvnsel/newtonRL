# cluster/pace_env.sh  --  source this in EVERY shell and EVERY job on PACE.
#
#   source /storage/project/r-jmcnabb3-0/luna/newtonRL/cluster/pace_env.sh
#
# Why this file exists: $HOME on Phoenix is 20 GB, and an Isaac Sim install
# plus its caches is comfortably more than that. Nothing here changes what the
# software does -- it only stops every component from writing into a home
# directory that cannot hold it. Skip it and you will fill your quota partway
# through a container pull, at which point the failure looks like a corrupt
# download rather than a full disk.
#
# The three filesystems, and what each is for:
#
#   home      20 GB, backed up.  Dotfiles and this repo's checkout at most.
#   project   1 TB, persistent.  Container image, conda env, the USD asset.
#              ^ 32 GB already in use by the group; it is SHARED, so keep an
#                eye on it and do not put run logs here.
#   scratch   15 TB, NOT backed up and PURGED on an access-time policy.
#              ^ logs, checkpoints, caches, particle dumps. Anything you would
#                be annoyed but not ruined to lose. Move finished runs to
#                project before the purge window.

# --- adjust these two if you lay things out differently -------------------
export LUNA_PROJECT="${LUNA_PROJECT:-/storage/project/r-jmcnabb3-0/luna}"
export LUNA_SCRATCH="${LUNA_SCRATCH:-/storage/scratch1/5/$USER/luna}"
# -------------------------------------------------------------------------

mkdir -p "$LUNA_PROJECT" "$LUNA_SCRATCH"/{cache,tmp,logs,runs}

# Apptainer: the image build and its scratch space. A pull unpacks layers
# before it assembles the .sif, so TMPDIR needs room for roughly twice the
# final image.
export APPTAINER_CACHEDIR="$LUNA_SCRATCH/cache/apptainer"
export APPTAINER_TMPDIR="$LUNA_SCRATCH/tmp/apptainer"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR"   # older name, same thing
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"

# Everything that respects the XDG spec -- which includes Warp's compiled
# kernel cache, and that one grows without bound across runs.
export XDG_CACHE_HOME="$LUNA_SCRATCH/cache/xdg"
export PIP_CACHE_DIR="$LUNA_SCRATCH/cache/pip"
export TMPDIR="$LUNA_SCRATCH/tmp"

# Omniverse and Isaac Sim ignore XDG and write to fixed paths under $HOME.
# These are the ones that actually fill a 20 GB quota.
export OMNI_CACHE_DIR="$LUNA_SCRATCH/cache/ov"
export OMNI_DATA_DIR="$LUNA_SCRATCH/cache/ov-data"
export ISAACSIM_CACHE_DIR="$LUNA_SCRATCH/cache/isaacsim"

mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$XDG_CACHE_HOME" \
         "$PIP_CACHE_DIR" "$OMNI_CACHE_DIR" "$OMNI_DATA_DIR" "$ISAACSIM_CACHE_DIR"

# Isaac Sim's EULA, which it will otherwise stop and ask about on a node with
# no terminal attached.
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y

export LUNA_REPO="${LUNA_REPO:-$LUNA_PROJECT/newtonRL}"
export LUNA_SIF="${LUNA_SIF:-$LUNA_PROJECT/isaaclab.sif}"

echo "[pace_env] project $LUNA_PROJECT"
echo "[pace_env] scratch $LUNA_SCRATCH"
echo "[pace_env] caches redirected off \$HOME (20 GB quota)"
