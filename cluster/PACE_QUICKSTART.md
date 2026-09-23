# Isaac Lab and Newton on PACE Cluster

---

## 0. Log in with username

```bash
ssh hzhang993@login-phoenix.pace.gatech.edu
```

And then:

```bash
unset SSH_ASKPASS SSH_ASKPASS_REQUIRE
export GIT_TERMINAL_PROMPT=1
```

PACE points `SSH_ASKPASS` at a GTK helper that cannot work over SSH, so
anything that prompts fails with "cannot open display" instead of asking.

---

## 1. Environment

```bash
mkdir -p /storage/scratch1/5/$USER/luna
cd /storage/scratch1/5/$USER/luna
export LUNA_SCRATCH=/storage/scratch1/5/$USER/luna
export LUNA_SIF=$LUNA_SCRATCH/isaaclab.sif
export XDG_CACHE_HOME=$LUNA_SCRATCH/cache/xdg
export APPTAINER_CACHEDIR=$LUNA_SCRATCH/cache/apptainer
export APPTAINER_TMPDIR=$LUNA_SCRATCH/tmp/apptainer
export TMPDIR=$LUNA_SCRATCH/tmp
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y PRIVACY_CONSENT=Y
mkdir -p $XDG_CACHE_HOME $APPTAINER_CACHEDIR $APPTAINER_TMPDIR $TMPDIR
```

---

## 2. Build the container (once, ~47 min)

Get a **CPU** node — this is a download and a compression, no GPU needed:

```bash
salloc -A gts-jmcnabb3 -N1 -n8 --mem=48G -t2:00:00
mkdir -p /tmp/sifbuild && cd /tmp/sifbuild
apptainer pull isaaclab.sif docker://nvcr.io/nvidia/isaac-lab:3.0.0-rc1
cp isaaclab.sif $LUNA_SCRATCH/isaaclab.sif
```

Build on `/tmp`, not scratch — mksquashfs on Lustre quoted 22 hours, `/tmp`
took 47 minutes. Pulls anonymously; no NGC account needed.

---

## 3. Get a GPU node for one hour

```bash
salloc -A gts-jmcnabb3 -q inferno -p gpu-rtx6000 -N1 --gres=gpu:1 -t1:00:00 --mem=64G
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

---

## 4. Copy the image to node-local disk

Re-run the step 1 exports first — `salloc` gives you a fresh shell and
`$LUNA_SCRATCH` will be empty otherwise.

```bash
cp $LUNA_SCRATCH/isaaclab.sif /tmp/isaaclab.sif
export SIF=/tmp/isaaclab.sif
```

---

## 5. Set up writable cache binds

```bash
export OV=$LUNA_SCRATCH/cache/ov-container
mkdir -p $OV/{kit-cache,kit-logs,kit-data,nv-omniverse,ov-cache}
export BINDS="-B $LUNA_SCRATCH:$LUNA_SCRATCH -B $OV/kit-cache:/isaac-sim/kit/cache -B $OV/kit-logs:/isaac-sim/kit/logs -B $OV/kit-data:/isaac-sim/kit/data -B $OV/nv-omniverse:/root/.nvidia-omniverse -B $OV/ov-cache:/root/.cache/ov"
```

---

## 6. Run the MPM granular demo

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/demos/mpm/newton_mpm_granular.py \
  --max_steps 200 --collider wedge --device cuda:0
```

A healthy run prints ~40 `OmniHub: Hub failed to launch` warnings, a protobuf
`File already exists in database` error, and `failed to open the default
display`. All normal. The `CUDA graph took: 40 s` line is one-time capture,
not per-step.

---


## 7. Get pictures out


```bash
apptainer exec $SIF cat \
  /workspace/isaaclab/scripts/demos/mpm/newton_mpm_granular.py \
  > $LUNA_SCRATCH/mpm_frames.py

# copy cluster/patch_demo_frames.py over too, then:
python3 patch_demo_frames.py $LUNA_SCRATCH/mpm_frames.py

FRAME_DIR=$LUNA_SCRATCH/frames FRAME_EVERY=10 \
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p $LUNA_SCRATCH/mpm_frames.py \
  --max_steps 200 --device cuda:0 --viz newton_gl

ls $LUNA_SCRATCH/frames
```

```bash
scp "hzhang993@login-phoenix.pace.gatech.edu:/storage/scratch1/5/hzhang993/luna/frames/*.png" .
```

---
