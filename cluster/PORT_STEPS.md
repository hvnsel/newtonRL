# Porting the excavation work into the upstream repo

- **SOURCE**: this checkout, `newtonRL`, branch `claude/great-shannon-mgcr7j`
- **TARGET**: the upstream repo, which has only the tricycle

The excavator package imports nothing from the tricycle, so TARGET's
tricycle files stay as they are. It does import `luna_hifi_tasks/ground.py`.

---

## Step 0: make a branch in TARGET

```bash
cd <target repo>
git checkout main
git pull
git checkout -b excavation
git status          # must say "nothing to commit, working tree clean"
```

---

## Step 1: copy the new files

Copy from SOURCE to the same paths in TARGET:

```
luna_hifi_tasks/excavator/       the whole package: the machine (excavator.py),
                                 shared config, terrain, mdp/, navigate/, excavate/
luna_hifi_tasks/ground.py        ground plane spawn config (if TARGET lacks it)
assets/ground_plane/             the ground plane USD it loads (if TARGET lacks it)
scripts/dig_demo.py              scripted dig, prints the fill readout
cluster/                         the whole directory: PACE setup, sbatch scripts,
                                 patch_demo_frames.py, this file
TUNING.md                        which knob lives in which file
```

Delete TARGET's root copies of `ClusterSetup.md` and `patch_demo_frames.py`;
the `cluster/` versions replace them.

Do not copy `assets/config.yaml` or `assets/.asset_hash` (converter output,
read by no code).

```bash
git status --short  # new files as ??, the two root files as D
```

---

## Step 2: edit `luna_hifi_tasks/__init__.py`

Add the excavator import, which registers its gym task ids:

```python
from . import tricycle  # noqa: F401
from . import excavator  # noqa: F401
```

---

## Step 3: ignore the generated excavator asset

The converted USD and its source stamp are built on each machine and not
tracked. Add to TARGET's `.gitignore`:

```
assets/excavator/
assets/excavator.usd.srchash
```

Then build the asset with the README steps from step 4.

---

## Step 4: edit `README.md`

Copy the section **"Rebuild the excavator asset (whenever `excavator.py`
changes)"** from SOURCE's README into TARGET's README at the same place: a
`###` inside `## 1. One-time setup`, after `### Convert the tricycle asset
(once)`.

Add the excavator rows to TARGET's own `## Repo layout` tree, keeping the rest
of TARGET's tree.

---

## Step 5: check registration

```bash
isaaclab -p -c "import luna_hifi_tasks, gymnasium; print([k for k in gymnasium.registry if k.startswith('Luna')])"
```

Expected: TARGET's tricycle id plus

```
'Luna-Excavator-Navigate', 'Luna-Excavator-Navigate-Course',
'Luna-Excavator-Excavate', 'Luna-Excavator-Excavate-Small', 'Luna-Excavator-Excavate-Micro'
```

---

## Step 6: commit

```bash
git add -A
git status          # new files; modified __init__.py, .gitignore, README.md; the two root files deleted
git commit -m "Add the lunar regolith excavation task"
```

---

## Not in this port

- `docs/tricycle_manual/`: the tricycle manual (LaTeX, PDF, figures).
- SOURCE's tricycle: a triangular-frame body with its regenerated USD and
  rewritten env files. It differs from TARGET's tricycle and is a separate
  port.
- The container image: `isaaclab.sif` (12 GB) lives in cluster scratch.
