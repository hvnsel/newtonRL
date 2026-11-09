# Training on PACE (Phoenix)

The same install as the workstation, without a container: Isaac Lab
`release/3.0.0` at the workstation's commit in a uv venv, kit-less (no Isaac
Sim), with the Newton MPM fix from `scripts/patch_newton_mpm.py`. Everything
lives in scratch, `/storage/scratch1/5/$USER/luna`; `$HOME` holds only 20 GB.

| file | what |
|---|---|
| `env.sh` | paths and caches; sourced by everything below |
| `setup_pace.sh` | the one-time install |
| `train_excavate.sbatch` | excavate, resuming from the newest checkpoint, up to `TOTAL` iterations |
| `train_navigate.sbatch` | navigate from scratch |

---

## 1. Log in and clone

```bash
ssh hzhang993@login-phoenix.pace.gatech.edu
mkdir -p /storage/scratch1/5/$USER/luna
git clone -b claude/great-shannon-mgcr7j https://github.com/hvnsel/newtonRL.git /storage/scratch1/5/$USER/luna/newtonRL
```

## 2. Install (once, on the login node)

Compute nodes may have no internet, so this runs on the login node.

```bash
bash /storage/scratch1/5/$USER/luna/newtonRL/cluster/setup_pace.sh
```

The last line prints the versions: `newton 1.6.1` and `warp 1.17.0`, as on
the workstation.

## 3. Copy the excavator asset from the workstation (once)

Building the asset needs Isaac Sim, which training never uses, so it is
copied. On the workstation:

```bash
scp -r ~/newtonRL/assets/excavator ~/newtonRL/assets/excavator.usd.srchash \
  hzhang993@login-phoenix.pace.gatech.edu:/storage/scratch1/5/hzhang993/luna/newtonRL/assets/
```

The stamp is a hash of the machine's MJCF, so the copy is valid as long as
`excavator.py` is the same on both.

## 4. Train excavate

Submit from the repo root. Every job appends to `dig_train.log` there.

```bash
source /storage/scratch1/5/$USER/luna/newtonRL/cluster/env.sh && cd $REPO
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

On the workstation, merging into its own logs:

```bash
scp -r hzhang993@login-phoenix.pace.gatech.edu:/storage/scratch1/5/hzhang993/luna/newtonRL/logs/rsl_rl/excavator_excavate ~/newtonRL/logs/rsl_rl/
```

## 6. Update the code

```bash
source /storage/scratch1/5/$USER/luna/newtonRL/cluster/env.sh && cd $REPO
git fetch origin claude/great-shannon-mgcr7j && git merge --ff-only origin/claude/great-shannon-mgcr7j
```

A change to `excavator.py`'s MJCF needs the asset rebuilt on the workstation
and copied again (step 3).

## 7. Navigate

```bash
source /storage/scratch1/5/$USER/luna/newtonRL/cluster/env.sh && cd $REPO
ENVS=1024 ITERS=4000 sbatch cluster/train_navigate.sbatch
```

---

## Notes

- Both sbatch files ask for an L40S (`--gres=gpu:l40s:1`).
  `sinfo -o "%.20P %.18G" | grep -i gpu` lists the other GPU types;
  `sbatch --gres=gpu:<type>:1 ...` overrides it per job.
- A shell that runs anything from this repo by hand needs
  `source $REPO/cluster/env.sh` first, for `$PY` and the caches.
- `uv sync` in `$ISAACLAB` again would reinstall Newton without the fix; rerun
  `$PY $REPO/scripts/patch_newton_mpm.py` after it.
