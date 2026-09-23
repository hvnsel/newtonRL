# Runbook: Isaac Lab + Newton MPM on PACE Phoenix, from zero

Everything below was run on PACE Phoenix on 2026-09-22 and worked. Substitute
your own username for `hzhang993` and your own charge account for
`gts-jmcnabb3`.

Nothing here needs a repo on the cluster, NGC credentials, or write access to
shared storage. It all lives in your personal scratch and `rm -rf` undoes it.

Total time from nothing: about **90 minutes**, of which 47 are an unattended
batch job you do not sit through.

---

## 0. Log in

```bash
ssh hzhang993@login-phoenix.pace.gatech.edu
```

Two things to do on first login, or they will bite later:

```bash
unset SSH_ASKPASS SSH_ASKPASS_REQUIRE
export GIT_TERMINAL_PROMPT=1
```

PACE points `SSH_ASKPASS` at a GTK helper that cannot work over SSH, so
anything that prompts fails with "cannot open display" instead of asking.
Worth putting in `~/.bashrc`; inside a batch job it otherwise blocks on an
invisible prompt and burns the whole walltime.

Also **paste one line at a time** on the login shell. It does not always strip
bracketed-paste escapes, so a multi-line paste arrives as
`$'\E[200~mkdir': command not found` with the rest running as arguments to
whatever survived.

---

## 1. Environment (every shell, every job)

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

`$HOME` is **20 GB** and Isaac Sim's caches exceed it. Omniverse and Warp both
ignore the usual cache variables and write to fixed paths under `$HOME`, so
skipping this fills the quota partway through the container pull, where it
looks like a corrupt download rather than a full disk.

(`cluster/pace_env.sh` in the repo does all of the above plus project-storage
fallback. The lines are inlined here so a teammate needs nothing but this
page.)

---

## 2. Build the container (once, ~47 min, unattended)

```bash
sbatch cluster/build_sif.sbatch      # or paste the two commands below
squeue -u $USER
tail -f isaaclab-sif-*.out
```

If you would rather do it by hand, get a **CPU** node -- this is a download
and a compression, and holding a GPU through it wastes credits:

```bash
salloc -A gts-jmcnabb3 -N1 -n8 --mem=48G -t2:00:00
cd /tmp && mkdir -p sifbuild && cd sifbuild
apptainer pull isaaclab.sif docker://nvcr.io/nvidia/isaac-lab:3.0.0-rc1
cp isaaclab.sif $LUNA_SCRATCH/isaaclab.sif
```

**Build on `/tmp`, not on scratch.** `mksquashfs` writes many small blocks and
Lustre is built for large sequential I/O; writing the image straight to
scratch quoted **22 hours**, and `/tmp` took **47 minutes** for a 12 GB image.
The finished `.sif` then moves to scratch as one sequential copy, which Lustre
is good at.

The image is marked Early Access on NGC but **pulls anonymously** -- no
account, no API key.

Expect a long silent stretch and a flood of `xattr` warnings during unpack.
Both are normal.

---

## 3. Get a GPU node

```bash
salloc -A gts-jmcnabb3 -q inferno -p gpu-rtx6000 -N1 --gres=gpu:1 -t1:00:00 --mem=64G
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Use `gpu-rtx6000` or `gpu-l40s`. Those have RT cores; a bare `--gres=gpu:1`
without `-p` lands on `gpu-v100`, which does not, and rendering is the part
that suffers there.

**Apptainer exists only on compute nodes.** On a login node `module avail`,
`module spider` and `command -v apptainer` all come back empty and look like a
definitive no. It is at `/usr/bin/apptainer` here.

---

## 4. Copy the image to node-local disk

```bash
cp $LUNA_SCRATCH/isaaclab.sif /tmp/isaaclab.sif
export SIF=/tmp/isaaclab.sif
```

Do not skip this. A run that died on a **command-line typo** still took
**5m09s**, all of it Apptainer reading 12 GB over Lustre -- and that cost is
paid on every single launch. One minute here makes everything after it start
in seconds. The copy dies with the node; the scratch original does not.

---

## 5. Set up the writable cache binds

```bash
export OV=$LUNA_SCRATCH/cache/ov-container
mkdir -p $OV/{kit-cache,kit-logs,kit-data,nv-omniverse,ov-cache}
export BINDS="-B $LUNA_SCRATCH:$LUNA_SCRATCH -B $OV/kit-cache:/isaac-sim/kit/cache -B $OV/kit-logs:/isaac-sim/kit/logs -B $OV/kit-data:/isaac-sim/kit/data -B $OV/nv-omniverse:/root/.nvidia-omniverse -B $OV/ov-cache:/root/.cache/ov"
```

Kit writes its shader and derived-data caches to fixed paths *inside* the
image, which is a read-only squashfs. Without these binds it still runs, but
logs `Failed to acquire exclusive lock to data store` and recompiles its
shaders on every launch.

---

## 6. Run the MPM granular demo

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/demos/mpm/newton_mpm_granular.py \
  --max_steps 200 --collider wedge --device cuda:0
```

Expected, after roughly two minutes of startup:

```
[INFO]: Isaac Lab Newton granular MPM demo ready. Spawned 48000 particles.
```

Three things that trip people up:

- **`--nv` is mandatory.** Without it a GPU node behaves exactly like a CPU
  node: `cuda available: False` and Kit dying on "Found no NVIDIA driver".
  It reads as a broken image and is not.
- **There is no `python` on `PATH`** inside the image -- Isaac Sim bundles its
  own interpreter. `apptainer exec image.sif python foo.py` fails with
  "executable file not found". Use `isaaclab.sh -p`.
- **`--headless` does not exist in 3.x.** It is `--viz`, taking
  `kit`, `newton_gl`, `newton_rtx`, `rerun`, `viser` or `none`. Headless is
  the default, reached by not naming a visualizer; `none` is the off switch
  and an empty string is rejected. `--video` and `--enable_cameras` are gone
  with it.

### Log lines that look like failures but are not

A healthy run prints all of these:

- `OmniHub: Hub failed to launch ...` x40, then `Not using Hub` and a
  successful fallback to a CloudFront mirror. Omniverse's asset CDN client,
  with nothing to connect to.
- `Extensions config 'extension.toml' doesn't exist .../tmp` -- Kit scanning
  the working directory for extension folders.
- `File already exists in database: grpc/health/v1/health.proto`.
- `failed to open the default display. Can't verify X Server version` --
  headless, as asked.
- `error calling pthread_setaffinity_np` -- the perf monitor wanting CPU
  pinning Slurm's cgroup will not give it.
- A `FutureWarning` about Newton shape colour replacement.

The `CUDA graph took: 40.2 s` line is a **one-time** graph capture, not a
per-step cost. A short run looks disproportionately slow because of it.

---

## 7. Verify the dependency stack (optional but fast)

```bash
# paste cluster/newton_smoke.py onto the cluster with `cat > newton_smoke.py`
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p newton_smoke.py
```

Six layers, each reporting independently: torch+GPU, warp, isaaclab,
`isaaclab_newton`, `isaaclab_contrib.coupling`, and a headless Kit start.
All six passed on `gpu-rtx6000` with driver 575.57.08.

---

## 8. Get pictures out

The demos have no `--video`. Isaac Lab 3.x does ship a video recorder, but the
*environment* base class builds it and `env.step()` drives it -- the demos
construct a simulation directly and never make an environment, so no flag
reaches it. Its frame source, `visualizer.render_rgb_array()`, is exposed on
every rendering visualizer, so a five-line patch to a copy of the demo writes
PNGs:

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

Look for `[FRAME] .../frame_00010.png (H, W, 3)`. If it instead prints
`no visualizer exposed render_rgb_array`, it names what *is* active -- switch
`--viz` to that.

`FRAME_EVERY=1` captures every step. Rendering every step is genuinely slow
and is what makes a short run look like a hang, so keep it high while
iterating and drop it to 1 only for a take you keep.

This is **offscreen rendering to disk via EGL** -- no X server, no display, no
forwarded port. It is not WebRTC and no streaming was tested.

Pull the frames back from your own machine:

```bash
scp "hzhang993@login-phoenix.pace.gatech.edu:/storage/scratch1/5/hzhang993/luna/frames/*.png" .
```

Scratch is shared Lustre, so the login node sees what the compute node wrote;
the allocation does not need to still be alive.

---

## Measured results

| | |
|---|---|
| image | `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`, 12 GB `.sif`, pulls anonymously |
| build | 47 min on `/tmp`; 22 h quoted on Lustre |
| isaaclab | 17.0.2 (the 3.x line) |
| torch / warp | 2.11.0+cu128 / 1.16.0, CUDA 12.9 |
| `isaaclab_newton` | present -- MPM and MJWarp solver configs |
| `isaaclab_contrib.coupling` | present -- `CouplerProxyCfg`, `CouplerEntryCfg` |
| GPU tested | Quadro RTX 6000, 25.2 GB, driver 575.57.08 |
| particles | **48,000** in `newton_mpm_granular.py` |
| Kit cold start | ~131 s |
| rendering | headless via EGL, frames to PNG |

One caveat to carry with the 48,000: that demo has a wedge and a ground plane,
so it measures MPM headroom with the rigid side nearly empty. It is not a
prediction of what a full excavator scene will hold.
