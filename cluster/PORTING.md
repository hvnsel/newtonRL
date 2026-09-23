# Porting the excavation work into the upstream repo

This checkout carries the tricycle task plus everything added for lunar
regolith excavation. Upstream has only the tricycle. The split is clean:

- **baseline** `86e523b` -- the last tricycle-only commit
- **43 files changed, 8451 insertions, 0 deletions**

Nothing under `luna_hifi_tasks/tricycle/`, `assets/` or `docs/` was touched,
so there is no risk of clobbering upstream's tricycle work.

## The one command

```bash
# in THIS checkout
git diff 86e523b HEAD > /tmp/excavation.patch

# in the upstream checkout, on a fresh branch
git checkout -b excavation
git apply --3way /tmp/excavation.patch
git status
```

`--3way` matters: it falls back to a real merge when a hunk does not apply
cleanly, and leaves conflict markers you can resolve instead of failing the
whole patch. Without it one drifted line rejects all 43 files.

If `git apply` reports failures, they will be in the four modified files
below -- the 39 new ones cannot conflict, because upstream has no version of
them.

## What is new (39 files, copy as-is)

```
luna_hifi_tasks/excavator/          the whole package: asset generator, cfg,
                                    mdp/{observations,rewards,sensors,terrain},
                                    navigate/ and excavate/ environments
scripts/check_excavator.py          offline geometry validator (MuJoCo, no Isaac Lab)
scripts/dig_demo.py                 scripted dig against the MPM bed
scripts/mpm_probe.py                empirical particle-capacity ladder
scripts/smoke_test.py               end-to-end env smoke test
scripts/train_budget.py             env-steps/s -> GPU hours per policy
tests/test_{observations,rewards,sensors,terrain}.py
cluster/                            PACE Phoenix setup: container build, env,
                                    dependency smoke test, frame capture, docs
```

## What is modified (4 files, all small)

`luna_hifi_tasks/__init__.py` -- one line. Without it the excavator tasks are
never registered and every `Luna-Excavator-*` gym id raises `NameNotFound`:

```python
from . import excavator  # noqa: F401
```

`pyproject.toml` -- an optional dependency group, so `check_excavator.py` can
run without Isaac Lab installed:

```toml
[project.optional-dependencies]
assets = ["mujoco>=3.13"]
```

The floor is not cosmetic. Below MuJoCo 3.13 `mj_geomDistance` silently
returns 0.0 for every pair, which turns every clearance and passage number
the checker prints into fiction.

`.gitignore` -- `assets/excavator/`, the converter's output. Note that
`assets/config.yaml` and `assets/.asset_hash` ARE tracked and the converter
rewrites them, so those two can drift out of step with the ignored directory.

`README.md` -- 72 added lines, no deletions.

## After applying

```bash
python -m pytest tests/ -q                    # the four new test modules
python scripts/check_excavator.py             # needs mujoco>=3.13
```

`check_excavator.py` runs offline and needs no GPU, no Isaac Lab and no USD,
so it is the fastest way to confirm the port landed intact.

## Caveat

The baseline `86e523b` is this checkout's last tricycle-only commit. If
upstream's tricycle has moved on since the fork, the four modified files may
need hand-merging -- which is why `--3way` is above, and why the split was
worth checking rather than assuming.
