# luna_hifi_tasks

Isaac Lab (develop / 3.0 beta, Newton backend) task package for rover excavation
on implicit-MPM soil.

> **These are the original template notes, written before anything ran.** The
> install commands, the `-v0` task ids and the six VERIFY items below are
> superseded by [RUNBOOK.md](RUNBOOK.md), which records the setup that actually
> trains and plays. Read the runbook first; keep this for the excavation
> package's STUB/VERIFY map, which is still accurate. For a line-by-line
> explanation of the tricycle task and how it plugs into Isaac Lab, Newton
> and rsl_rl, see `docs/tricycle_manual/main.pdf`.

```
Luna_HiFi/                  <- pyproject.toml lives here; pip install -e this folder
  pyproject.toml
  luna_hifi_tasks/
    __init__.py
    tricycle/               <- START HERE. Pipeline validation, ~1k particles/env.
      __init__.py           gym.register
      tricycle.py           the car as an MJCF string (1 steer wheel, 2 driven)
      tricycle_env_cfg.py   flat soil strip, 2 actions, 15 obs, 2 s episodes
      tricycle_env.py       reward = forward distance
      agents/__init__.py
      agents/rsl_rl_ppo_cfg.py
    excavation/             the real task. Everything the tricycle taught you applies.
      __init__.py
      excavation_env_cfg.py scene, physics, spaces, reward weights   <- most STUB/VERIFY here
      excavation_env.py     DirectRLEnv: actions, obs, reward, dones, reset
      soil.py               SoilMaterial mapping + height-map / bucket-fill kernels
      agents/__init__.py
      agents/rsl_rl_ppo_cfg.py
```

## Install

From the Isaac Lab root, with Isaac Lab develop already set up:

```
./isaaclab.bat -p -m pip install -e C:\path\to\luna_hifi_tasks
./isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task Luna-Tricycle-Direct-v0 --num_envs 64 --headless
```

Isaac Lab's train script only sees tasks whose package has been imported. If
`--task` says the id is unknown, add `import luna_hifi_tasks` next to the
existing task imports at the top of train.py / play.py, or use the
`--task` entry from a package Isaac Lab discovers via its extension mechanism.

## Why the tricycle exists

It answers three questions before the rover costs you a week: does Isaac Lab
develop + isaaclab_newton + MPM run on this machine; do the VERIFY items below
resolve; does reward go up. Expect mean episode reward (metres in 2 s) to climb
toward 3-4 m within a few hundred iterations on 64 envs. If it stays at zero,
the wheels aren't turning (actuator/joint-name mismatch) or the car is falling
through the strip (particles-per-cell too low, or MPM cfg not attached).

## Two kinds of placeholder

`STUB` — a value or function you replace with your own: USD path, joint regexes,
actuator gains, bucket cavity box, reward terms, success threshold.

`VERIFY` — an `isaaclab_newton` API name I could not confirm against the develop
branch. There are six, all at import or attribute-access sites:

1. `MPMSolverCfg`, `MPMObjectCfg`, `MPMGridCfg`, `MPMParticleMaterialCfg` import paths
2. How `NewtonCfg` holds an MJWarp solver **and** an MPM solver at once
   (`mpm_solver_cfg=` is a guess; copy what the Franka-Pour task cfg does)
3. `MPMGridCfg` field names (`size`, `particles_per_cell`, `material`)
4. `MPMObject.data.particle_pos_w` / `.particle_mass` attribute names and layout
5. `MPMObject.reset(env_ids)` — and that it clears solver history, not just positions
6. `sim_utils.MjcfFileCfg` in Kit-less Newton mode (tricycle only). If it doesn't
   spawn, convert `tricycle.xml` to USD once and switch to `UsdFileCfg`.

Resolve all five by reading one file: the Franka-Pour task under `isaaclab_tasks`
(or `isaaclab_contrib`). It exercises every one of them.

## What carries over from the standalone scripts

`SoilMaterial` is the same class. `to_material_cfg()` replaces
`emit_soil()`'s `custom_attributes` dict. Nothing else from `newton_soil.py`
is needed — solver construction, coupling, stepping, and `project_outside`
are all done by `isaaclab_newton`'s MPM manager.

## Order of operations once it imports

1. `--num_envs 1` and confirm the rover spawns, the bed spawns, arm moves.
2. Print `_bucket_fill` while driving the arm by hand (random policy). If it never
   goes non-zero, the cavity box is wrong or the particle layout assumption is.
3. Only then scale envs. MPM cost is dominated by active grid cells × envs.
