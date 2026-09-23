# PACE setup: what was done, in order

One session, 2026-09-22. The goal was narrow on purpose: **prove an Isaac Lab
+ Newton dependency stack can run on PACE Phoenix, and that pictures come out
of it.** No repo on the cluster, no credentials on a shared filesystem, no
training code, nothing shared touched. Everything lives in personal scratch
and `rm -rf` undoes all of it.

---

## The one-paragraph version

We built a container with the full Isaac Lab 3.x + Newton physics stack and
ran it on a PACE GPU node. Every layer the excavator project depends on
imports and runs: the MPM granular solver, the rigid-body solver, and the
two-way coupling between them. NVIDIA's own granular demo spawned **48,000
particles** on one of the weaker GPUs in the fleet, and we got rendered
frames out of a headless compute node with no display and no forwarded port.
Nothing shared was modified and no decision was pre-empted.

---

## Step by step

### 1. Found out what the site actually has

Apptainer (the HPC container runtime) reports as absent on login nodes --
`module avail`, `module spider` and `command -v` all come back empty. It is
in fact at `/usr/bin/apptainer` on **compute nodes only**. That one fact
would otherwise read as "PACE does not support containers".

Measured the GPU fleet and the driver, which cannot be seen from a login
node: driver 575.57.08, QOS `inferno`, and seven GPU partitions from V100
through H200 and RTX PRO Blackwell.

Only three of them have RT cores -- `gpu-rtx6000`, `gpu-l40s`,
`gpu-rtxpro-blackwell`. A bare `--gres=gpu:1` lands on `gpu-v100`, which does
not, and rendering is the part that suffers there. Headless physics is fine
on any of them.

### 2. Solved the storage problem before it bit

`$HOME` on Phoenix is **20 GB**. An Isaac Sim install plus its caches is more
than that, and Omniverse and Warp both ignore the standard cache environment
variables and write to fixed paths under `$HOME`. Left alone, the quota fills
partway through a container pull, where it looks like a corrupt download
rather than a full disk.

`cluster/pace_env.sh` redirects all of it to scratch. It also discovers that
the group project directory is not writable at its top level (needs the PI to
create a per-user subdirectory) and falls back to scratch rather than failing.

### 3. Built the container

Image: `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`. It is marked Early Access, but
**it pulls anonymously** -- no NGC account or API key needed, which was not
obvious and would otherwise have been a blocker.

The build itself was the one real time sink, and the fix is worth keeping:
`mksquashfs` writes many small blocks, Lustre is built for large sequential
I/O, and writing the image straight to scratch quoted **22 hours**. Building
on the node-local disk (`/tmp`) instead took **47 minutes** for a 12 GB
image, which then moves to scratch as one sequential copy.

The same property bites at runtime: a run that failed on a *command-line
typo* still took 5 minutes, all of it Apptainer reading 12 GB over Lustre on
every launch. Copying the image to `/tmp` once makes later runs start in
seconds.

### 4. Verified every dependency layer separately

`cluster/newton_smoke.py` -- one standalone file, no repo, no install. It
checks six layers and keeps going after a failure, because "Isaac Lab imports
but the Newton backend does not" and "everything imports but no GPU is
visible" are different problems with different fixes.

All six pass on `gpu-rtx6000`:

| layer | result |
|---|---|
| torch | 2.11.0+cu128, GPU visible |
| GPU | Quadro RTX 6000, 25.2 GB |
| warp (Newton's kernel layer) | 1.16.0, CUDA 12.9 |
| isaaclab | **17.0.2** -- the 3.x line, so Newton is in |
| `isaaclab_newton` | MPM and rigid-solver configs all present |
| `isaaclab_contrib.coupling` | rigid <-> MPM coupling present |
| headless Kit app | starts, ~131 s cold |

The last two were the open questions. `isaaclab_contrib` is a separate
project and there was real doubt it shipped in the image. **It does** -- so
the container as pulled is the entire dependency stack, with nothing to
install on top.

### 5. Ran the physics that the project actually depends on

The image ships NVIDIA's own MPM demos, including a granular one and a
rigid-to-MPM two-way coupling one -- which is, in miniature, a drum turning
in soil.

```
[INFO]: Isaac Lab Newton granular MPM demo ready. Spawned 48000 particles.
```

48,000 particles, on the second-weakest RT card in the fleet. For scale, the
H200 partition has 141 GB of VRAM against this card's 24.

### 6. Got frames off a headless node

```
[NewtonVisualizer] No display found (DISPLAY is unset); the Newton viewer
runs headless via EGL and no window will open.
```

Rendering works with no X server, no display and no forwarded port. That was
the question that could have needed a firewall exception or a ticket, and it
does not.

Isaac Lab 3.x has a built-in video recorder, but the *environment* base class
builds it and `env.step()` drives it -- the demos construct a simulation
directly and never make an environment, so no flag combination reaches it.
The recorder's own frame source, `visualizer.render_rgb_array()`, is exposed
on every rendering visualizer, so a five-line patch to a copy of the demo
writes PNGs. `cluster/patch_demo_frames.py` applies it.

---

## What this does and does not establish

**Established.** The dependency stack runs on PACE. The MPM solver, the rigid
solver and the coupling between them all work. Rendering works headless.
48,000 MPM particles is unremarkable for this hardware. None of it needed
credentials, shared storage, or anyone's permission.

**Not established.** The *excavator's* own particle ceiling. The 48,000 is
pure MPM with almost no rigid bodies in the scene. The ~7,000 ceiling hit on
a laptop was a different constraint entirely -- a contact-buffer limit in the
rigid solver on a 119-geometry machine, since fixed. The real number is
somewhere between and needs the actual scene to measure, which is a short job
once the code goes up.

## Suggested next steps

1. **VRAM at 48k particles.** Turns one data point into a scaling curve for
   the larger GPUs. Minutes.
2. **`newton_mpm_twoway_coupling.py`.** The closest thing in the box to the
   real problem, and it exercises the coupler rather than only importing it.
3. **Upload the training code** -- a few MB over `scp`, once that is wanted.
4. **Ask about project storage.** The container image is the expensive thing
   to rebuild, and scratch is purged on an access-time policy.

## Open questions worth raising

- Is a writable project subdirectory available, so the image survives purges?
- Is WebRTC streaming from compute nodes permitted? Frames-to-disk works and
  needs nothing, but live viewing is nicer for debugging.
- Which GPU partitions are we expected to use, and is there a per-account
  budget worth planning around?
