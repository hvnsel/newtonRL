# Training on PACE (Phoenix)

Isaac Lab `release/3.0.0` (commit `961760bc`) in a uv venv, kit-less (no
Isaac Sim), with Newton 1.6.1 plus four soil-solver files from 1.7.0 and one
fix, installed and checked by `scripts/patch_newton_mpm.py`. Everything
lives in scratch, `/storage/scratch1/5/$USER/luna`; `$HOME` holds only 20 GB.

Below, `<user>` is your GT username and `<repo>` is where this repository is
cloned on PACE, e.g. `/storage/scratch1/5/<user>/luna/Luna_HiFi`.

| file | what |
|---|---|
| `env.sh` | paths and caches; sourced by everything below |
| `setup_pace.sh` | the one-time install |
| `train_excavate.sbatch` | excavate, resuming from the newest checkpoint, up to `TOTAL` iterations |
| `train_navigate.sbatch` | navigate from scratch |
| `play_mpm.sbatch` | one episode of the RFT-trained policy on MPM soil, with video, trace and GIF |
| `stubs/omni/client.py` | a stand-in for `omni.client`, whose wheel needs a newer glibc than PACE has; `env.sh` puts it on `PYTHONPATH` |

---

## 1. Log in and clone

```bash
ssh <user>@login-phoenix.pace.gatech.edu
mkdir -p /storage/scratch1/5/$USER/luna
git clone <repository URL> <repo>
```

## 2. Install (once, on the login node)

Compute nodes may have no internet, so this runs on the login node.

```bash
bash <repo>/cluster/setup_pace.sh
```

Near the end it prints `all 10 files match: Newton is set up for excavate`;
if any file differs, the setup stops there. The last line prints the
versions, `warp 1.17.0` among them. `newton.__version__` can read 1.6.1 or
1.7.0 depending on how Newton was installed; only the label differs.

Set up before this check existed? Update the code (step 6), then:

```bash
source <repo>/cluster/env.sh && $PY $REPO/scripts/patch_newton_mpm.py
```

## 3. Copy the excavator asset (once)

Building the asset needs Isaac Sim, which this install leaves out. Build it
on a machine set up with `workstation/Setup.md` (README.md, "Assets"), then
copy it from that machine's repo folder:

```bash
scp -r assets/excavator assets/excavator.usd.srchash <user>@login-phoenix.pace.gatech.edu:<repo>/assets/
```

The stamp is a hash of the machine's MJCF, so the copy is valid as long as
`excavator.py` is the same on both.

## 4. Train excavate

Submit from the repo root. Every job appends to `dig_train.log` there.

```bash
source <repo>/cluster/env.sh && cd $REPO
sbatch cluster/train_excavate.sbatch
```

A job is cut off at 12 hours and checkpoints every 5 iterations. Queue three
in a row; each resumes from the newest checkpoint, and one that finds
`TOTAL` reached exits at once:

```bash
j=$(sbatch --parsable cluster/train_excavate.sbatch); for i in 2 3; do j=$(sbatch --parsable --dependency=afterany:$j cluster/train_excavate.sbatch); done
```

`TOTAL=300` sets the target iteration (default 200); `FRESH=1` starts a new
run instead of resuming.

Check on it:

```bash
squeue -u $USER
grep -a "\[excavate\] tool" dig_train.log | tail -1
grep -a -E "Learning iteration|Iteration time|pass_done|payload_kg" dig_train.log | tail -4
scancel <jobid>                     # stop a job
```

The first job compiles Warp's kernels into `$LUNA/cache/warp`, so its start
is slower than later ones.

## 5. Bring checkpoints back for playback

From the repo folder on the machine you play back on:

```bash
scp -r <user>@login-phoenix.pace.gatech.edu:<repo>/logs/rsl_rl/excavator_excavate logs/rsl_rl/
```

## 6. Update the code

```bash
source <repo>/cluster/env.sh && cd $REPO
git pull
```

A change to `excavator.py`'s MJCF needs the asset rebuilt and copied again
(step 3).

## 7. Navigate

```bash
source <repo>/cluster/env.sh && cd $REPO
ENVS=1024 ITERS=4000 sbatch cluster/train_navigate.sbatch
```

## 8. The RFT policy on MPM soil

Copy the asset (step 3) and an RFT checkpoint into the same path under
`<repo>`, from the repo folder on the machine that trained it:

```bash
R=logs/rsl_rl/excavator_excavate_rft/<run>
ssh <user>@login-phoenix.pace.gatech.edu mkdir -p <repo>/$R
scp $R/model_<N>.pt <user>@login-phoenix.pace.gatech.edu:<repo>/$R/
```

Then submit from the repo root. The job plays one episode with the cut set
to a -0.10 slope from 5 cm and writes `dig_mpm_4cm.{csv,npz,gif,mp4}` there;
every job appends to `dig_mpm_play.log`.

```bash
source <repo>/cluster/env.sh && cd $REPO
sbatch cluster/play_mpm.sbatch
VOXEL=0.03 sbatch -t 1:15:00 cluster/play_mpm.sbatch    # finer grid, dig_mpm_3cm.*
VIDEO=0 sbatch cluster/play_mpm.sbatch       # if the node cannot render the video
```

The video needs moviepy in the venv; `setup_pace.sh` installs it. An install
from before that gets it with `uv pip install --python $PY -e "$REPO[video]"`.

MPM's drum contact reaches half a voxel beyond the drum: 2 cm on the 4 cm
grid, 4 cm on the 8 cm default, which the policy never met in training.

---

## Notes

- The train sbatch files ask for an L40S (`--gres=gpu:l40s:1`); `play_mpm.sbatch` takes any GPU.
  `sinfo -o "%.20P %.18G" | grep -i gpu` lists the other GPU types;
  `sbatch --gres=gpu:<type>:1 ...` overrides it per job.
- A shell that runs anything from this repo by hand needs
  `source <repo>/cluster/env.sh` first, for `$PY` and the caches.
- `uv sync` in `$ISAACLAB` again can reinstall Newton without these changes;
  rerun `setup_pace.sh` after it. Without them excavate stops when the soil
  solver is built (`RuntimeError: Keys and values array storage must be
  large enough to contain 2*count elements`).
- PACE's glibc (2.34) is too old for the `omniverseclient` and `ovstage`
  wheels, so `setup_pace.sh` skips both. Isaac Lab still calls `omni.client`
  for any file a USD references that is not on disk; the ground plane's
  `OmniPBR.mdl` material is one. Without `stubs/omni/client.py` on
  `PYTHONPATH`, the excavate env stops at startup with
  `ModuleNotFoundError: No module named 'omni'`.
