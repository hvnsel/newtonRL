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

## 8. Clone the repo

Into scratch, not `$HOME` -- `$HOME` is 20 GB and Omniverse writes there
regardless of `XDG_CACHE_HOME`.

```bash
cd $LUNA_SCRATCH
git clone https://github.com/<owner>/Luna_HiFi.git
cd Luna_HiFi
export REPO=$LUNA_SCRATCH/Luna_HiFi
```

For a private repo git asks for a username and password over HTTPS; the
password is a GitHub personal access token with `repo` scope, not the account
password. The `GIT_TERMINAL_PROMPT=1` and `unset SSH_ASKPASS` from step 0 are
what let it ask at all.

To avoid retyping the token every pull, without writing it into `.git/config`
where a `git remote -v` would print it:

```bash
git config --global credential.helper 'cache --timeout=36000'
```

`assets/excavator/` is committed, so the clone carries the converted USD and
nothing here needs the MJCF converter or `isaacsim`.

Clear the staleness guard. A clone writes every file at checkout time, so
`excavator.py` and `excavator.usda` land within the same second and
`usd_status()` compares them with a strict `>` -- whether it fires is down to
file ordering:

```bash
touch $REPO/assets/excavator/excavator.usda
```

---

## 9. Install the package into the container

The `.sif` is read-only, so the install goes to a writable directory on
scratch that is bound into the container. `--user` puts it in
`$PYTHONUSERBASE`, which Python adds to `sys.path` on its own:

```bash
export PYTHONUSERBASE=$LUNA_SCRATCH/pyuser
mkdir -p $PYTHONUSERBASE
export BINDS="$BINDS -B $PYTHONUSERBASE:$PYTHONUSERBASE"

apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p -m pip install --user --no-deps -e $REPO
```

`--no-deps` because the image already has torch, warp and rsl_rl at the
versions its solvers were built against, and resolving `dependencies = []`
against PyPI can only disturb that.

`-m pip`, not a bare `pip`: the image has no `python` or `pip` on `PATH`.

Both `PYTHONUSERBASE` and the repo must stay bound on every later `exec`, or
the import resolves to nothing.

Check it registered. This needs the entry point from `pyproject.toml` as well
as the import, so it tests the port's two config edits at once:

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p -c \
  "import luna_hifi_tasks, gymnasium as gym; print(luna_hifi_tasks.__file__); print([k for k in gym.registry if 'Luna-' in k])"
```

Expect the path under `$REPO` and five ids: the tricycle and the four
excavator tasks. No excavator ids means `luna_hifi_tasks/__init__.py` is
missing its `from . import excavator`.

---

## 10. Run

Smoke test first -- builds the scene and steps it with no policy, so it
separates a scene problem from a training problem:

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p $REPO/scripts/run_task.py \
  --task Luna-Excavator-Excavate-Micro --num_envs 1 --steps 200
```

Then navigate, which is the rigid tier and carries no MPM grid:

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Luna-Excavator-Navigate --num_envs 64 --max_iterations 10 \
  physics=newton_mjwarp
```

Then excavate, starting small. The MPM grid is a step function in env count --
`next_pow2(8 * envs * 161001) * 192` bytes, so 26 envs is 6.4 GB and 52 is
12.9 GB. `gpu-rtx6000` has 24 GB, so 26 is the ceiling there; `max_num_envs`
is set to 104 for an 80 GB H100:

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Luna-Excavator-Excavate --num_envs 8 --max_iterations 10 \
  physics=newton_mjwarp
```

Logs land in the working directory, so `cd $REPO` first if you want them
under the repo rather than wherever you launched from.

Measure throughput before committing to an env count:

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p $REPO/scripts/dig_demo.py
```

Sections 8 to 10 are written from the repo and the container's documented
behaviour, not from a run on PACE. Sections 0 to 7 were measured.

---

## Reference

Measured on PACE Phoenix, 2026-09-22, `gpu-rtx6000`, driver 575.57.08.

| | |
|---|---|
| image | `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`, 12 GB `.sif`, pulls anonymously |
| build | 47 min on `/tmp`; 22 h quoted on Lustre |
| isaaclab | 17.0.2 |
| torch / warp | 2.11.0+cu128 / 1.16.0, CUDA 12.9 |
| `isaaclab_newton` | present -- MPM and MJWarp solver configs |
| `isaaclab_contrib.coupling` | present -- `CouplerProxyCfg`, `CouplerEntryCfg` |
| particles | 48,000 in `newton_mpm_granular.py` |
| Kit cold start | ~131 s |
| rendering | headless via EGL |

The 48,000 is MPM headroom with almost no rigid bodies in the scene. The
excavator's own ceiling is set by the rigid solver's contact buffers and has
not been measured on this hardware.

### Things that are true and not discoverable

- **Apptainer is on compute nodes only**, at `/usr/bin/apptainer`. On a login
  node `module avail`, `module spider` and `command -v` all come back empty.
- **`$HOME` is 20 GB** and Omniverse and Warp write to fixed paths under it,
  ignoring `XDG_CACHE_HOME`. A full quota surfaces as a corrupt download.
- **`--nv` is mandatory.** Without it a GPU node reports `cuda available:
  False` and Kit dies on "Found no NVIDIA driver".
- **There is no `python` on `PATH`** in the image. Use
  `/workspace/isaaclab/isaaclab.sh -p`, which warns that it is deprecated in
  favour of `uv run isaaclab` from 3.1.
- **`--headless` does not exist in 3.x.** `--viz` takes `kit`, `newton_gl`,
  `newton_rtx`, `rerun`, `viser` or `none`; headless is the default and `none`
  is the off switch. `--video` and `--enable_cameras` are gone with it.
- **Kit's caches live inside the read-only image.** Unbound, every launch
  recompiles its shaders and logs `Failed to acquire exclusive lock to data
  store`.
- Only `gpu-rtx6000`, `gpu-l40s` and `gpu-rtxpro-blackwell` have RT cores. A
  bare `--gres=gpu:1` lands on `gpu-v100`, which does not.
