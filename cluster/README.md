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
  python /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py \
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

```bash
apptainer pull "$LUNA_SIF" docker://nvcr.io/nvidia/isaac-lab:<TAG>
```

Two things to settle first, both by checking rather than assuming:

- **the tag.** It must be an Isaac Lab **3.x with the Newton backend**. Newton
  does not exist on 2.x, and layer 4 of the smoke test below is exactly the
  check for this.
- **NGC auth.** Some NVIDIA images need a key:
  `apptainer remote login --username '$oauthtoken' docker://nvcr.io`

If no published tag carries Newton, the fallback is an Apptainer definition
file starting from the Isaac Sim base image with Isaac Lab's develop branch
installed on top. More work, but it is the honest answer if the tag does not
exist -- and worth finding out before building anything around it.

## 4. Prove the stack works

`newton_smoke.py` is standalone: no repo, no package install, no credentials.
Copy the one file over and run it in the container.

```bash
apptainer exec --nv -B "$LUNA_SCRATCH" "$LUNA_SIF" python newton_smoke.py
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
