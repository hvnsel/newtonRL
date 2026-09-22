# Newton on PACE Phoenix: dependencies only

The goal here is a **container with a working Isaac Lab + Newton stack**, and
nothing else. No repo on the cluster, no GitHub credentials on a shared
filesystem, no training code until it is wanted. When the time comes the
training code is a few MB and `scp` handles it in seconds.

That ordering is also the cheaper one: the dependency stack is the part that
takes a day to get right, and it can be validated with a single standalone
file.

## Day 0: the 30-minute tentative test

Before any of the below. The question is only *"can we get an Isaac Sim
environment running on PACE and see pictures come out of it"*, and answering it
touches nothing shared and commits to nothing.

Nothing needs to be copied from your machine. Paste files in with `cat >`.
Everything lands in your own scratch, which no one else can see, and
`rm -rf` undoes all of it.

**1. Is something already provided?** Cheapest possible check -- PACE may ship
Isaac Sim or a container for it, which saves a 20 GB pull:

```bash
module avail 2>&1 | grep -iE "isaac|omniverse|apptainer|singularity"
ls /storage/coda1/shared 2>/dev/null | head
```

**2. Get a GPU for half an hour.** Interactive, so you see errors as they
happen, and it costs a fraction of a credit:

```bash
salloc -A gts-jmcnabb3 -N1 --gres=gpu:1 -t0:30:00
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

That driver version is the one number that can veto an Isaac Sim image, and it
is invisible from the login node.

**3. Get the container.** In the job, after `source pace_env.sh`.

**4. Does the stack work?** `newton_smoke.py`, layered so a failure says which
layer.

**5. Do frames come out?** Isaac Lab ships its own demos, so no scene code
needs writing:

```bash
apptainer exec --nv -B "$LUNA_SCRATCH" "$LUNA_SIF" \
  /workspace/isaaclab/isaaclab.sh -p \
    /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Cartpole-v0 --headless --video --video_length 200 \
    --max_iterations 5
```

`--headless --video` renders offscreen and writes an mp4 under the run's log
directory. Pull it back with `scp` from your own machine and watch it. That is
the whole "can we see what the sim is doing on a cluster" question answered,
with a task that ships in the box.

WebRTC streaming is the nicer version of this and worth asking about at the
meeting -- but frames-to-disk needs no open ports, no firewall exceptions and
nobody's permission, so it is the right thing to have working first.

**Nothing here is a commitment.** No repo on the cluster, no credentials, no
shared directories touched, no long jobs queued. If the meeting sends things a
different way, `rm -rf /storage/scratch1/5/$USER/luna` and it never happened.

---

## The storage rule that governs everything

| | size | persists? | use for |
|---|---|---|---|
| `$HOME` | **20 GB** | backed up | dotfiles, nothing else |
| `/storage/project/r-jmcnabb3-0` | 1 TB, **shared** | yes | the `.sif`, if the PI grants you a subdirectory |
| `/storage/scratch1/5/$USER` | 15 TB | **purged** on access time | everything else |

An Isaac Sim install plus its caches does not fit in 20 GB, and Omniverse and
Warp both ignore `XDG_CACHE_HOME` and write to fixed paths under `$HOME`.
`pace_env.sh` redirects all of it. Skip it and the quota fills partway through
a container pull, where it looks like a corrupt download rather than a full
disk.

The group project directory is usually not writable by members at its top
level -- `mkdir` there returns "Permission denied" until the PI creates you a
subdirectory. `pace_env.sh` detects that and falls back to scratch, saying so.
Worth asking for project space eventually, for one reason: **the `.sif` is the
expensive thing to rebuild**, and scratch is purged on an access-time policy
that `noatime` mounts can defeat.

## 0. First-login gotchas

Paste **one line at a time**. The login shell does not always strip bracketed
paste escapes, so a multi-line paste arrives as `$'\E[200~mkdir': command not
found` and the rest runs as arguments to whatever survived.

And PACE sets `SSH_ASKPASS` to a GTK helper that cannot work over SSH:

```bash
unset SSH_ASKPASS SSH_ASKPASS_REQUIRE
export GIT_TERMINAL_PROMPT=1
```

Worth putting in `~/.bashrc` -- it otherwise bites inside batch jobs too, where
a job blocking on an invisible prompt burns its whole walltime.

## 1. What the site offers

```bash
bash pace_probe.sh
```

Partitions, GPU types and the exact GRES strings to request them, whether
Apptainer is a module, your charge accounts. All site-specific; none of it
worth assuming.

The one thing it cannot see from a login node is the GPU driver version, which
is the single most likely reason an Isaac Sim image refuses to run:

```bash
salloc -A gts-jmcnabb3 -N1 --gres=gpu:1 -t0:20:00
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

### What PACE actually has (measured 2026-09-22)

| partition | GPU | VRAM | RT cores? | nodes x GPUs |
|---|---|---|---|---|
| `gpu-v100` | V100 | 16 GB | **no** (Volta) | 40x2 |
| `gpu-rtx6000` | RTX 6000 | 24 GB | yes (Turing) | 30x4 |
| `gpu-a100` | A100 | 40-80 GB | **no** | 12x2, 1x8 |
| `gpu-l40s` | L40S | 48 GB | yes (Ada) | 10x8 |
| `gpu-h100` | H100 | 80 GB | **no** | 4x8 |
| `gpu-h200` | H200 | 141 GB | **no** | 12x8 |
| `gpu-rtxpro-blackwell` | RTX PRO 6000 | 96 GB | yes (Blackwell) | 3x8 |

Driver 575.57.08, QOS `inferno`.

A bare `--gres=gpu:1` lands on `gpu-v100`, which is the one partition whose GPU
has **no RT cores**. Isaac Sim's documented requirement is an RTX-class GPU, so
rendering is the part at risk there -- headless physics is not.

Split the work:

- **rendering, video, anything visual** -> `gpu-l40s`, the part NVIDIA actually
  targets for Omniverse. `gpu-rtx6000` is the fallback and is less contended.
- **headless MPM training** -> `gpu-h200` or `gpu-a100` for memory and compute.

```bash
salloc -A gts-jmcnabb3 -p gpu-l40s -N1 --gres=gpu:l40s:1 -t0:30:00
```

## 2. Environment

```bash
source pace_env.sh
```

Every shell, every job. It prints where it put things and which storage it
chose.

## 3. The container

Isaac Lab is run from a container on clusters: a native install wants write
access to paths a shared filesystem will not give you, and pins a driver you do
not control.

Confirmed on PACE: `apptainer` lives at `/usr/bin/apptainer` on **compute
nodes only** -- it is absent from login nodes and there is no module for it, so
`module avail` and `command -v` on a login node both come back empty and look
like a definitive no. It is not.

The image is `nvcr.io/nvidia/isaac-lab:3.0.0-rc1` (NGC, 15.2 GB compressed,
3.0 line so it carries Newton). It is marked Early Access, so it needs NGC
credentials:

```bash
# key from ngc.nvidia.com -> Setup -> Generate API Key
apptainer remote login --username '$oauthtoken' docker://nvcr.io
apptainer pull "$LUNA_SIF" docker://nvcr.io/nvidia/isaac-lab:3.0.0-rc1
```

**Pull on a CPU node.** A 15 GB download plus unpack takes far longer than it
takes to establish anything, and there is no reason to hold a GPU while it
happens:

```bash
salloc -A gts-jmcnabb3 -N1 -t2:00:00        # no --gres
```

`APPTAINER_TMPDIR` needs room for roughly twice the final image while layers
are assembled.

**Build on node-local disk, not on scratch.** `mksquashfs` writes many small
blocks and Lustre is built for large sequential I/O; writing the image straight
to scratch quoted **22 hours**. On Phoenix the node-local disk is `/tmp`, and
the same build there took **47 minutes** for a **12 GB** image, which then
moves to scratch as one sequential copy. `build_sif.sbatch` does this, finds
the local disk itself, and runs unattended.

No NGC credentials were needed: `3.0.0-rc1` pulls anonymously despite the Early
Access label.

## 4. Prove the stack works

`newton_smoke.py` is standalone: no repo, no package install, no credentials.
Copy the one file over and run it in the container.

```bash
apptainer exec --nv -B "$LUNA_SCRATCH" "$LUNA_SIF" \
  /workspace/isaaclab/isaaclab.sh -p newton_smoke.py
```

It checks six layers and keeps going where it can, because "isaaclab imports
but isaaclab_newton does not" and "everything imports but no GPU is visible"
are different problems with different fixes:

1. torch, and whether a GPU is actually visible
2. warp, its devices, its kernel cache location
3. isaaclab
4. **isaaclab_newton** -- MPM and MJWarp configs. The 3.x check.
5. `isaaclab_contrib.coupling` -- rigid <-> MPM, which the excavator needs
6. a headless Kit app actually starting, which is where the driver and the
   EULA get their say

Run it on a **GPU node**, not the login node. Layers 1-5 will pass on a login
node and tell you nothing about the thing most likely to be wrong.

`--nv` is what bind-mounts the host driver into the container. Without it a
GPU node reports exactly what a CPU node reports -- `cuda available: False`,
and layer 6 dying on "Found no NVIDIA driver on your system" -- which reads as
a broken image rather than a missing flag.

### All six layers pass (measured 2026-09-22, `gpu-rtx6000`, driver 575.57.08)

| layer | result |
|---|---|
| torch | 2.11.0+cu128, `cuda available: True` |
| GPU | Quadro RTX 6000, sm_75, 25.2 GB |
| warp | 1.16.0, CUDA Toolkit 12.9 / Driver 12.9, `cuda:0` with mempool |
| isaaclab | **17.0.2** -- the 3.x line, so Newton is in |
| `isaaclab_newton` | `MPMObjectCfg`, `MPMSolverCfg`, `NewtonCfg`, `MJWarpSolverCfg`, `MPMParticleMaterialCfg` |
| `isaaclab_contrib.coupling` | `CouplerProxyCfg`, `CouplerEntryCfg` |
| headless Kit | starts, 110.3.0, ~131 s cold |

Layers 4 and 5 were the two open questions and both are answered: the NGC
image ships `isaaclab_contrib` as well as `isaaclab_newton`, so **nothing has
to be pip-installed into an overlay**. The container as pulled is the whole
dependency stack.

Its Python is 3.12.13 and the container reports `Linux ... el9_6`, so the
libraries are the image's, not the host's -- the only thing that comes from
PACE is the driver, via `--nv`.

### Log lines on a successful start that look like failures

A clean Kit launch in this container prints all of these. None of them mean
anything is wrong, and between them they account for the first two minutes:

- `OmniHub: Hub failed to launch ... retry_reason` x40. Omniverse's asset-CDN
  client, with no Hub in the image and no Nucleus server to reach. It backs
  off and gives up, and Kit continues.
- `Extensions config 'extension.toml' doesn't exist .../luna/tmp` -- Kit
  scanned the working directory for extension folders and found the `tmp/`
  and `cache/` that `pace_env.sh` creates.
- `File already exists in database: grpc/health/v1/health.proto` -- protobuf
  double-registration, present in every Isaac Sim run.
- `failed to open the default display. Can't verify X Server version` --
  headless, as asked.
- `error calling pthread_setaffinity_np` -- the perf monitor wanting CPU
  pinning that Slurm's cgroup will not give it.

Two more are real, but non-fatal, and **only appear when the caches are not
bound**:

```
[Error] [omni.datastore] Failed to acquire exclusive lock to data store (256>=256)
[Error] [omni.datastore] Failed to create local file data store at '/isaac-sim/kit/cache/DerivedDataCache'
[Error] [omni.kit.app.plugin] failed to open file '/isaac-sim/kit/data/Kit/IsaacLab/3.0/user.config.json'
```

Both paths are **inside the read-only squashfs**. Kit shrugs and runs without
a cache, which means every single launch recompiles its shaders and pays the
full cold start. `run_in_container.sh` binds writable scratch over them.

### Running anything: `run_in_container.sh`

```bash
source cluster/pace_env.sh
bash cluster/run_in_container.sh newton_smoke.py
```

It supplies `--nv`, the bundled interpreter, and the cache binds -- the three
things that are easy to omit and that each fail without naming themselves.
Note that Warp respects `XDG_CACHE_HOME` but only if it is exported *before*
`apptainer`: a run without `pace_env.sh` puts the Warp kernel cache in
`$HOME/.cache/warp`, where it grows without bound against a 20 GB quota.

### Inside the image

There is no `python` on `PATH` -- Isaac Sim bundles its own interpreter, so
`apptainer exec image.sif python foo.py` fails with "executable file not found"
and looks like a broken container. It is not.

```
/workspace/isaaclab            Isaac Lab root
/workspace/isaaclab/isaaclab.sh -p   <- the Linux twin of isaaclab.bat -p
/isaac-sim/python.sh                 <- Isaac Sim's own launcher
/isaac-sim/kit/python/bin/python3    <- the bare interpreter, env not set up
```

Prefer `isaaclab.sh -p`: it wires up the extension paths and environment that a
bare interpreter does not.

It warns that it is **deprecated and removed in Isaac Lab 3.1**, in favour of
`uv run isaaclab`. It still resolves the right interpreter today, so this is a
shelf life rather than a problem -- but anything written against it needs
revisiting at the 3.1 bump, and `run_in_container.sh` is the single place that
has to change.

The demo scripts also moved in the 3.x layout, so paths copied from 2.x docs
(`/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py`) are not
there. Find them rather than guess:

```bash
apptainer exec "$LUNA_SIF" bash -lc 'ls /workspace/isaaclab/scripts'
```

### The demos worth running: `scripts/demos/mpm/`

The 3.x image ships MPM demos, which are a far better proof than a cartpole
because they exercise the exact subsystem the excavator stands on:

```
scripts/demos/mpm/newton_mpm_granular.py         granular MPM -- the soil
scripts/demos/mpm/newton_mpm_twoway_coupling.py  rigid <-> MPM -- drum in soil
scripts/demos/mpm/snowball_smash.py
scripts/demos/mpm/teapot_fill.py
scripts/demos/newton_viewer_dominoes.py
scripts/demos/newton_viewer_block_and_tackle.py
```

Running those answers "does the granular physics we need work on this
cluster", not merely "does Isaac Sim start". They are NVIDIA's own code, so a
failure in them is a site or image problem and never ours -- which makes them
the right first thing to run on any new node type.

Note `scripts/reinforcement_learning/` exists but contains no `train.py`: the
RL entry point was renamed in 3.x as well as moved, so check the directory
rather than porting a 2.x command line.

### `--headless` no longer exists

This is the 3.x change that breaks every command line copied from a 2.x doc,
including the `--headless --video` recipe for pulling frames. AppLauncher now
takes:

```
--visualizer VIS, --viz VIS   CSV of backends: kit, newton, rerun, viser
--experience FILE             resolved FROM the visualizer when left empty
```

Headless is the default, reached by *not* naming a visualizer rather than by
asking for headlessness. `--enable_cameras` and `--video` are gone with it, so
frame capture is no longer a launcher flag -- it is whatever the chosen
visualizer backend does. `rerun` and `viser` are the two to investigate; viser
is a web server and so needs SSH port forwarding, which makes it the more
awkward of the two on a cluster.

The MPM demos take `--max_steps N`, which bounds a run without a visualizer
and makes them usable as a timed throughput benchmark:

```bash
bash cluster/run_in_container.sh \
  /workspace/isaaclab/scripts/demos/mpm/newton_mpm_granular.py \
  --max_steps 200 --collider wedge --device cuda:0
```

`--voxel_size` is exposed there too, which is the same knob that decides
whether soil can enter the excavator drum at all.

Anything that starts Kit also wants writable cache directories, and the image
is read-only, so bind scratch and point the Omniverse cache variables at it --
`pace_env.sh` sets them.

## 5. Only then, the training code

```bash
scp -r newtonRL hzhang993@login-phoenix.pace.gatech.edu:/storage/scratch1/5/$USER/luna/
```

It is a few MB. The generated USD asset is not in it -- build that in the
container once, with the converter.

## Numbers not to carry over from a laptop

- `nconmax` / `njmax` are **fixed-size per-env buffers** in the rigid solver.
  Overrunning one is an illegal memory access, not an error: a CUDA 700 storm,
  or 0xC0000374 on Windows. This cost about five rounds of misdiagnosis.
- the MPM sparse-grid caps are **absolute totals across all envs** and do not
  scale with `--num_envs`.
- `voxel_size` decides whether soil can enter the drum at all. The coupler eats
  a whole voxel out of every passage and particles sit one voxel apart, so the
  drum's 0.096 m entry channel is 3.8 grain diameters at 0.02 and 2.2 at 0.03.
  Granular material arches below about four.
