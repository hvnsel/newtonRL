# Porting the excavation work into the upstream repo

Slow version. Two repos are involved:

- **SOURCE** -- this checkout, `newtonRL`, branch `claude/great-shannon-mgcr7j`
- **TARGET** -- the upstream repo, which today has only the tricycle

The excavator package is self-contained: every import inside
`luna_hifi_tasks/excavator/` and `scripts/` resolves within
`luna_hifi_tasks.excavator.*`. Nothing in it imports the tricycle or
`soil_simulations`, so no tricycle file has to change for excavation to run.

29 files are brand new, 4 existing files each gain a small block, and 2 files
at the TARGET root move into `cluster/`.

**Before you start, read "The tricycle has diverged" at the bottom.** SOURCE's
tricycle is not TARGET's tricycle. That is a separate decision from this port,
and this port does not require you to make it.

---

## Step 0 -- make a branch in the TARGET repo

Do not do this on `main`. If something goes wrong you want one command to undo
it.

```bash
cd <target repo>
git checkout main
git pull
git checkout -b excavation
```

Check you are clean before you start:

```bash
git status
```

It should say "nothing to commit, working tree clean". If it does not, commit
or stash whatever is there first.

---

## Step 1 -- copy the 29 new files

These do not exist in TARGET, so there is nothing to merge. Copy them wholesale,
keeping the same paths.

**The excavator package** -- copy the entire directory:

```
luna_hifi_tasks/excavator/
```

It contains 20 files:

```
__init__.py                      registers the four gym task ids
excavator.py                     the machine, as MJCF. The geometry lives here
excavator_cfg.py                 shared ArticulationCfg: actuators, spawn pose
excavator_env_base.py            what navigate and excavate share
terrain_cfg.py                   procedural terrain for the rigid tier
mdp/__init__.py
mdp/observations.py              declared observation specs
mdp/rewards.py                   reward and termination terms, pure functions
mdp/sensors.py                   soil heightmap and drum fill
mdp/terrain.py                   one scan interface, two backends
navigate/__init__.py
navigate/navigate_env.py
navigate/navigate_env_cfg.py
navigate/agents/__init__.py
navigate/agents/rsl_rl_ppo_cfg.py
excavate/__init__.py
excavate/excavate_env.py
excavate/excavate_env_cfg.py
excavate/agents/__init__.py
excavate/agents/rsl_rl_ppo_cfg.py
```

**Four scripts** into a `scripts/` directory. TARGET has no `scripts/`
directory, so create it:

```
scripts/check_excavator.py       offline geometry check. MuJoCo, no Isaac Lab
scripts/run_task.py              build and step either task, no policy
scripts/dig_demo.py              scripted dig, prints the fill readout
scripts/reward_audit.py          per-episode reward totals under a fixed policy
```

**Four cluster files** into a new `cluster/` directory:

```
cluster/ClusterSetup.md          PACE setup, start to finish
cluster/patch_demo_frames.py     adds PNG capture to an Isaac Lab MPM demo
cluster/build_sif.sbatch         builds the container unattended
cluster/PORT_STEPS.md            this file
```

TARGET has `patch_demo_frames.py` and `ClusterSetup.md` at its **root**.
Delete both root copies and use the `cluster/` ones -- these are the newer
versions, and they belong together.

**One doc** at the TARGET root:

```
TUNING.md                        which knob lives in which file
```

Do **not** copy `assets/config.yaml` or `assets/.asset_hash`. They are
converter droppings with absolute Windows paths baked in, read by no code, and
rewritten the next time the converter runs.

Confirm the copy landed:

```bash
git status --short
```

You should see 29 lines starting with `??`, and 2 starting with `D` -- the two
root files that moved into `cluster/`.

---

## Step 2 -- edit `luna_hifi_tasks/__init__.py`

This is the important one. Without it, nothing else works.

The file currently reads:

```python
from . import tricycle  # noqa: F401
```

Add one line so it reads:

```python
from . import tricycle  # noqa: F401
from . import excavator  # noqa: F401
```

**What this does.** Importing `luna_hifi_tasks.excavator` is what runs the
`gym.register(...)` calls inside `excavator/navigate/__init__.py` and
`excavator/excavate/__init__.py`. Without this line the four task ids are never
registered and every command fails with:

```
gymnasium.error.NameNotFound: Environment Luna-Excavator-Excavate doesn't exist.
```

---

## Step 3 -- edit `pyproject.toml`

Add this block. It goes after `dependencies = []` and before
`[project.entry-points."isaaclab.tasks"]`:

```toml
# scripts/check_excavator.py validates the excavator geometry offline and needs
# MuJoCo but deliberately NOT Isaac Lab, so it runs wherever you are editing
# excavator.py.
[project.optional-dependencies]
assets = ["mujoco>=3.13"]       # below this mj_geomDistance silently returns 0.0
                                # for every pair, which makes every clearance and
                                # passage number in check_excavator.py a fiction
```

**What this does.** Nothing at runtime -- it declares the dependency
`check_excavator.py` needs, so `pip install -e ".[assets]"` pulls it.

The `>=3.13` floor matters. On older MuJoCo, `mj_geomDistance` returns 0.0 for
every pair of geoms, so every clearance reads 0.0000 and the checker reports
failures on a model that is fine.

---

## Step 4 -- edit `.gitignore`

Append:

```
# Generated by the MJCF -> USD converter from luna_hifi_tasks/excavator/excavator.py.
# assets/config.yaml and assets/.asset_hash are tracked and rewritten by the
# converter, so they can drift out of step with this directory.
# scripts/dig_demo.py and scripts/run_task.py check the asset against its
# source before spawning anything.
assets/excavator/
```

**What this does.** `assets/excavator/` holds the converted USD, which is
generated from `excavator.py` and is large. It should not be committed.

---

## Step 5 -- edit `README.md`

Copy the section titled **"Rebuild the excavator asset (whenever
`excavator.py` changes)"** from SOURCE's README -- lines 122 to 189, a `###`
heading inside `## 1. One-time setup`. It sits between
`### Convert the tricycle asset (once)` and `## 2. Daily commands`. Paste it
in the same place in TARGET.

68 lines, appended only -- no existing README text changes.

Then add the excavator rows to TARGET's own `## Repo layout` tree. Do not
replace that tree with SOURCE's: SOURCE's lists no `soil_simulations/` and
describes the redesigned tricycle.

---

## Step 6 -- check it before you commit

Run these in TARGET. Each one is fast and each one proves a different thing.

**6a. Does the package import and register?** Needs Isaac Lab:

```bash
isaaclab -p -c "import luna_hifi_tasks, gymnasium; print([k for k in gymnasium.registry if k.startswith('Luna')])"
```

Expected -- five ids, one tricycle and four excavator:

```
['Luna-Tricycle-v0', 'Luna-Excavator-Navigate', 'Luna-Excavator-Excavate',
 'Luna-Excavator-Excavate-Small', 'Luna-Excavator-Excavate-Micro']
```

If the excavator ids are missing, step 2 did not take.

**6b. Is the geometry intact?** Needs MuJoCo only, no GPU, no Isaac Lab, so it
runs anywhere:

```bash
pip install "mujoco>=3.13"
python scripts/check_excavator.py
```

Expected, on the last line:

```
  all checks passed
```

Some numbers to sanity-check against, because a silently truncated file would
still run:

```
channel between the two lips  0.1083 .. 0.1429 m centreline
  -> opening 0.0963 m, +0.0463 m clear after the coupler margin
covered by the inner lip   100%
covered by the outer lip   100%
vs wheels min gap +0.1337 m
```

**6c. Does an env actually build?** Needs a GPU:

```bash
isaaclab -p scripts/run_task.py --task Luna-Excavator-Excavate-Micro --num_envs 1 --steps 200
```

This one needs the converted USD, so if you have not run the converter in
TARGET yet it will stop and tell you how. That is the expected failure, not a
port problem.

---

## Step 7 -- commit

```bash
git add -A
git status          # 26 new files, 4 modified. Nothing deleted.
git commit -m "Add the lunar regolith excavation task"
```

If `git status` shows anything **deleted**, stop -- something went wrong in
step 1 and you have overwritten a tricycle file.

---

## If you would rather not copy by hand

The same change as one patch:

```bash
git checkout -b excavation
git apply --3way excavation.patch
git status
```

`--3way` matters. Without it, one line that has drifted in TARGET rejects all
30 files instead of leaving you a conflict to resolve. Conflicts, if any, will
only be in the four files from steps 2-5; the 26 new files cannot conflict
because TARGET has no version of them.

---

## The tricycle has diverged

SOURCE and TARGET are not parent and child. Since the commit they share,
SOURCE has:

- redesigned the tricycle body as a triangular frame and regenerated its USD
  (`b532b33`), which rewrote `assets/tricycle/payloads/base.usda`,
  `payloads/Physics/mujoco.usda` and `payloads/Physics/physics.usda`
- rewritten `tricycle.py`, `tricycle_env.py`, `tricycle_env_cfg.py`,
  `tricycle/__init__.py` and `tricycle/agents/rsl_rl_ppo_cfg.py`
- deleted `luna_hifi_tasks/tricycle/soil.py`
- deleted `README_RL.md` and the root `__init__.py`

and TARGET has a `soil_simulations/` directory that SOURCE never had.

**None of that is required for excavation.** Porting the tricycle changes is a
separate decision, with its own risk: it changes the machine every existing
tricycle checkpoint was trained against. Do the excavation port first, confirm
it runs, then decide about the tricycle on its own merits.

---

## What is NOT in this port

- **The converted USD.** `assets/excavator/` is generated. Run the converter
  in TARGET once, following the README section from step 5. Note that
  `usd_status()` in `excavator_cfg.py` compares **mtimes**: a fresh clone
  stamps `excavator.py` at checkout time, so it will read as newer than any
  USD you copy in and the guard will refuse to run. Rebuild, or `touch` the
  USD once you are sure the geometry matches.

- **The tricycle manual.** `docs/tricycle_manual/` is 23 files of LaTeX, a
  built PDF and figures, all about the tricycle. Copy it only if you want it.

- **The tricycle changes.** See "The tricycle has diverged" above.
- **The container image.** `isaaclab.sif` is 12 GB and lives in cluster
  scratch, not in git.
- **Unit tests.** There are none for the excavator MDP terms. `rewards.py`,
  `observations.py` and `sensors.py` are unpinned, which matters because a
  sign error in a reward does not crash -- it trains smoothly toward the wrong
  thing.
