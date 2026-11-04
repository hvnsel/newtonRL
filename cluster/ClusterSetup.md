# Isaac Lab and Newton on PACE Cluster

---

## 0. Log in

```bash
ssh hzhang993@login-phoenix.pace.gatech.edu
unset SSH_ASKPASS SSH_ASKPASS_REQUIRE      # git prompts in the terminal
export GIT_TERMINAL_PROMPT=1
```

---

## 1. Environment

Run in every new shell, including the one `salloc` opens.

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

On a CPU node, building on node-local `/tmp`:

```bash
salloc -A gts-jmcnabb3 -N1 -n8 --mem=48G -t2:00:00
mkdir -p /tmp/sifbuild && cd /tmp/sifbuild
apptainer pull isaaclab.sif docker://nvcr.io/nvidia/isaac-lab:3.0.0-rc1
cp isaaclab.sif $LUNA_SCRATCH/isaaclab.sif
```

The image pulls anonymously. Once the repo is cloned (step 8),
`sbatch cluster/build_sif.sbatch` does the same as a batch job.

---

## 3. Get a GPU node for one hour

```bash
salloc -A gts-jmcnabb3 -q inferno -p gpu-rtx6000 -N1 --gres=gpu:1 -t1:00:00 --mem=64G
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

---

## 4. Copy the image to node-local disk

Re-run the step 1 exports in this shell, then:

```bash
cp $LUNA_SCRATCH/isaaclab.sif /tmp/isaaclab.sif
export SIF=/tmp/isaaclab.sif
```

---

## 5. Set up writable cache binds

Binds Kit's cache paths inside the read-only image to scratch.

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

Harmless output: ~40 `OmniHub: Hub failed to launch` warnings, a protobuf
`File already exists in database` error, and `failed to open the default
display`. `CUDA graph took: 40 s` is a one-time capture.

---

## 7. Save demo frames as PNGs

Copy `cluster/patch_demo_frames.py` to the cluster, then:

```bash
# extract the demo from the image
apptainer exec $SIF cat \
  /workspace/isaaclab/scripts/demos/mpm/newton_mpm_granular.py \
  > $LUNA_SCRATCH/mpm_frames.py

# add PNG capture to it
python3 patch_demo_frames.py $LUNA_SCRATCH/mpm_frames.py

# run it; writes every 10th frame to $LUNA_SCRATCH/frames
FRAME_DIR=$LUNA_SCRATCH/frames FRAME_EVERY=10 \
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p $LUNA_SCRATCH/mpm_frames.py \
  --max_steps 200 --device cuda:0 --viz newton_gl

ls $LUNA_SCRATCH/frames
```

From your own machine:

```bash
scp "hzhang993@login-phoenix.pace.gatech.edu:/storage/scratch1/5/hzhang993/luna/frames/*.png" .
```

---

## 8. Clone the repo

Into scratch:

```bash
cd $LUNA_SCRATCH
git clone https://github.com/<owner>/Luna_HiFi.git
cd Luna_HiFi
export REPO=$LUNA_SCRATCH/Luna_HiFi
```

For a private repo git prompts for a username and password; the password is
a GitHub personal access token with `repo` scope. Cache it for 10 hours:

```bash
git config --global credential.helper 'cache --timeout=36000'
```

---

## 9. Install the package into the container

`--user` installs into `$PYTHONUSERBASE` on scratch, bound into the
container. `--no-deps` leaves the image's torch, warp and rsl_rl untouched.

```bash
export PYTHONUSERBASE=$LUNA_SCRATCH/pyuser
mkdir -p $PYTHONUSERBASE
export BINDS="$BINDS -B $PYTHONUSERBASE:$PYTHONUSERBASE"

apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p -m pip install --user --no-deps -e $REPO
```

Every later `apptainer exec` needs `PYTHONUSERBASE` exported and bound.

Check the tasks are registered:

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p -c \
  "import luna_hifi_tasks, gymnasium as gym; print(luna_hifi_tasks.__file__); print([k for k in gym.registry if 'Luna-' in k])"
```

Prints a path under `$REPO`, `Luna-Tricycle-Direct` and the
`Luna-Excavator-*` ids.

---

## 10. Build the excavator asset

The converted excavator USD is not in git. Write the MJCF, remove the old
output, and convert to `assets/excavator.usd` (the converter writes
`assets/excavator/excavator.usda`):

```bash
apptainer exec --nv $BINDS $SIF /workspace/isaaclab/isaaclab.sh -p -c \
  "from luna_hifi_tasks.excavator.excavator import write_mjcf; print(write_mjcf())"
rm -rf $REPO/assets/excavator
apptainer exec --nv $BINDS $SIF /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/tools/convert_mjcf.py <path printed above> $REPO/assets/excavator.usd
```

---

## 11. Run

Logs are written under the current directory:

```bash
cd $REPO
```

Navigate (rigid only, no MPM grid):

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Luna-Excavator-Navigate --num_envs 64 --max_iterations 10 \
  physics=newton_mjwarp
```

Excavate. `--num_envs` is capped by `max_num_envs` in
`excavate/excavate_env_cfg.py` (4):

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p \
  /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Luna-Excavator-Excavate --num_envs 4 --max_iterations 10 \
  physics=newton_mjwarp
```

Scripted dig, no policy; prints the drum fill and throughput:

```bash
apptainer exec --nv $BINDS $SIF \
  /workspace/isaaclab/isaaclab.sh -p $REPO/scripts/dig_demo.py
```

Unattended navigate training:

```bash
sbatch cluster/train_navigate.sbatch              # as its own job
bash cluster/run_navigate_bg.sh runA              # detached, inside an allocation you hold
```

---

## Reference

PACE Phoenix, `gpu-rtx6000`, driver 575.57.08.

| | |
|---|---|
| image | `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`, 12 GB `.sif` |
| isaaclab | 17.0.2 |
| torch / warp | 2.11.0+cu128 / 1.16.0, CUDA 12.9 |
| `isaaclab_newton` | MPM and MJWarp solver configs |
| `isaaclab_contrib.coupling` | `CouplerProxyCfg`, `CouplerEntryCfg` |
| Kit cold start | ~131 s |
| rendering | headless via EGL |

### Notes

- Apptainer is at `/usr/bin/apptainer` on compute nodes only; login nodes
  have none.
- `$HOME` has a 20 GB quota. Omniverse and Warp write to fixed paths under
  it, ignoring `XDG_CACHE_HOME`; a full quota shows up as a corrupt download.
- `--nv` exposes the GPU to the container. Without it Kit fails with "Found no
  NVIDIA driver".
- The image has no `python` or `pip` on `PATH`; use
  `/workspace/isaaclab/isaaclab.sh -p`.
- 3.x has no `--headless`. `--viz` takes `kit`, `newton_gl`, `newton_rtx`,
  `rerun`, `viser` or `none`; headless is the default.
- Kit's caches are inside the read-only image; the step 5 binds redirect them.
- `gpu-rtx6000`, `gpu-l40s` and `gpu-rtxpro-blackwell` have RT cores. A bare
  `--gres=gpu:1` can land on `gpu-v100` (compute capability 7.0), which the
  image does not support (it needs >= 7.5).
