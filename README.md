# luna_hifi — Isaac Lab + Newton MPM tricycle task

A minimal, **working** Isaac Lab task that runs an articulated vehicle and
implicit-MPM soil in the same simulation, trained with rsl_rl PPO.

Validated end to end as of 2026-09-03 on Windows 11, RTX A1000 6 GB laptop:
the tricycle trains, plays, and the soil particles are visible.

Isaac Lab **develop** branch (3.0 line), Newton backend.
API reference: <https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/api/index.html>

For a line-by-line explanation of every file and how it plugs into Isaac Lab,
Newton and rsl_rl, see `docs/tricycle_manual/main.pdf`.

---

## Why the tricycle exists

It answers three questions cheaply, before a real task costs you a week: does
Isaac Lab develop + `isaaclab_newton` + MPM run on this machine; does rigid ↔
MPM coupling actually work; does reward go up.

Reward is metres travelled in a 2 s episode. If it stays at zero, the wheels
aren't turning (actuator / joint-name mismatch) or the car is falling through
the strip (particles-per-cell too low, or the MPM entry not coupled).

---

## Repo layout

```
luna_hifi/
  pyproject.toml                    declares the isaaclab.tasks entry point
  luna_hifi_tasks/
    __init__.py                     imports subpackages -> triggers gym.register
    tricycle/
      __init__.py                   gym.register("Luna-Tricycle-Direct")
      tricycle.py                   the car as an MJCF string (1 steer wheel, 2 driven)
      tricycle_env_cfg.py           scene, physics, coupling, spaces
      tricycle_env.py               DirectRLEnv: actions, obs, reward, dones, reset
      agents/rsl_rl_ppo_cfg.py      PPO runner cfg
  assets/tricycle/tricycle.usda     converted car asset (regenerate after editing tricycle.py)
  docs/tricycle_manual/             the line-by-line manual (LaTeX source + PDF)
```

Expected layout on disk, with Isaac Lab as a sibling:

```
C:\Users\hanse\Documents\
    IsaacLab\                       framework, develop branch, kit-less + isaacsim pip pkg
        .venv\                      Python 3.12 venv — everything runs through this
    luna_hifi\                      THIS REPO
```

---

## 1. One-time setup

### Python 3.12 (3.10 does NOT work)

Newton 1.5 annotates with `wp.array[wp.bool] | None`. Python 3.10's
`typing._type_check` rejects that; 3.11+ doesn't. Also imgui-bundle's newest
cp310 Windows wheel is 1.5.2, far below newton's `>=1.92.0` floor, so pip falls
back to a source build that fails.

```powershell
winget install --id Python.Python.3.12 -e
# reopen PowerShell
py -0                                   # confirm -V:3.12 is listed
```

### Isaac Lab

```powershell
cd C:\Users\hanse\Documents
git clone -b develop https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
.\isaaclab.bat -i
.\isaaclab.bat -i isaacsim        # needed ONLY for the MJCF->USD converter
```

If pip dies with `[Errno 13] Permission denied` on its wheel cache:
`python -m pip cache purge`, or `$env:PIP_NO_CACHE_DIR = "1"`.

### Verify Isaac Lab alone

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 physics=newton_mjwarp --viz newton
```

A pyglet `AssertionError` after "Training time:" is a harmless window-teardown
race. Ignore it.

### Install this package

```powershell
.\isaaclab.bat -p -m pip install -e C:\Users\hanse\Documents\luna_hifi
.\isaaclab.bat -p -c "import luna_hifi_tasks, gymnasium as gym; print([k for k in gym.registry if 'Luna-' in k])"
```

Must print `['Luna-Tricycle-Direct']`. Empty list = registration broken, see
Troubleshooting.

### Convert the tricycle asset (once)

```powershell
mkdir C:\Users\hanse\Documents\luna_hifi\assets
.\isaaclab.bat -p -c "from luna_hifi_tasks.tricycle.tricycle import write_mjcf; print(write_mjcf())"
.\isaaclab.bat -p scripts\tools\convert_mjcf.py C:\Users\hanse\AppData\Local\Temp\luna_hifi_assets\tricycle.xml C:\Users\hanse\Documents\luna_hifi\assets\tricycle.usd
```

Note: it writes `assets\tricycle\tricycle.usda` (subfolder, `.usda`), not the
path you gave. `tricycle_env_cfg.py` points at the real output. Redo this
whenever `tricycle.py` changes.

---

### Rebuild the excavator asset (whenever `excavator.py` changes)

`excavator.py` is the source; `assets\excavator\excavator.usda` is what
training loads. They only agree if you re-run this, and nothing warns you when
they don't — the old USD just keeps loading.

Do NOT hardcode the repo path. There is more than one `luna_hifi*` checkout on
this machine and the tricycle's lives somewhere else; converting into the wrong
one writes a USD that nothing loads while the env keeps reading the stale asset
from the right one. Derive it from the installed package instead, which is by
construction the same root `EXCAVATOR_USD_PATH` resolves against:

```powershell
# Plain python, NOT isaaclab.bat: the launcher writes an "[INFO] Using Python:"
# line on another stream that lands unpredictably in the captured output, so
# any positional Select-Object on it eventually picks up the INFO line instead
# of the answer. And find_spec LOCATES the package without executing it, so
# this needs neither Isaac Lab nor a Kit app.
$REPO = (python -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('luna_hifi_tasks').origin)))").Trim()

$REPO                                    # sanity: is this the newtonRL checkout?
git -C $REPO remote -v                   # sanity: does it point at hvnsel/newtonRL?
Test-Path $REPO\scripts\dig_demo.py      # sanity: must be True
```

If the package is not installed, or you want to find every checkout on the
machine rather than the installed one:

```powershell
Get-ChildItem C:\Users\hanse -Directory -Recurse -Depth 3 -ErrorAction SilentlyContinue |
  Where-Object { Test-Path (Join-Path $_.FullName "scripts\dig_demo.py") } |
  Select-Object -ExpandProperty FullName
```

Then:

```powershell
# 1. Check the geometry BEFORE converting. Needs MuJoCo, not Isaac Lab.
.\isaaclab.bat -p -m pip install mujoco
.\isaaclab.bat -p $REPO\scripts\check_excavator.py

# 2. Write the MJCF.
.\isaaclab.bat -p -c "from luna_hifi_tasks.excavator.excavator import write_mjcf; print(write_mjcf())"

# 3. Delete the old output first. The converter appends _1 rather than
#    overwriting, and the env then loads the STALE asset from the old folder.
Remove-Item -Recurse -Force $REPO\assets\excavator -ErrorAction SilentlyContinue

# 4. Convert. Note the output path: assets\excavator.usd, NOT
#    assets\excavator\excavator.usd -- the converter keeps only the directory
#    and forces <stem>\<stem>.usda, so the nested path lands a level too deep
#    and the spawn fails with FileNotFoundError.
.\isaaclab.bat -p scripts\tools\convert_mjcf.py $env:TEMP\luna_hifi_assets\excavator.xml $REPO\assets\excavator.usd

# 5. Watch it dig.
.\isaaclab.bat -p $REPO\scripts\dig_demo.py
```

`check_excavator.py` is step 1 for a reason. It catches the class of fault that
survives conversion and then never gets reported: MuJoCo does not test a body
against its own parent, and Isaac articulations default to
`self_collision=False`, so an arm passing through its own drum runs perfectly
happily in both and only shows up as a policy that will not learn. It measures
swept envelopes off the geoms, ray-probes the drum cavity, and sweeps the arms
through `ARM_RANGE` against the wheels and frame.

---

## 2. Daily commands

Every session, in each new PowerShell window:

```powershell
cd C:\Users\hanse\Documents\IsaacLab
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Train:

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Luna-Tricycle-Direct --num_envs 4
```

Play a trained checkpoint (soil particles visible):

```powershell
.\isaaclab.bat play --rl_library rsl_rl --task Luna-Tricycle-Direct --num_envs 1 --checkpoint latest --viz newton
```

Watch training live: add `--viz newton`. Logs and checkpoints land in
`logs\rsl_rl\tricycle\<timestamp>\`, saved every 50 iterations.

---

## 3. Hard-won facts

**CLI shape.** Not `scripts/reinforcement_learning/rsl_rl/train.py` — that
script doesn't exist on develop. It's `isaaclab train|play|zero_agent|random_agent`.
No `--headless` flag; headless is the default. `--viz`, not `--visualizer`.
Task IDs carry no `-v0` suffix. Backend override is `physics=newton_mjwarp`
(a Hydra config group), not `presets=newton`.

**External task discovery** is via the `isaaclab.tasks` entry point in
`pyproject.toml`. Entry points bake in at install time — after editing them,
reinstall with `-e`. A generated `TricycleDemo`-style scaffold is not needed;
entry points make it redundant.

**Editable install: when a reinstall is actually needed.** Only after editing
`pyproject.toml`. Verified on setuptools 79 with a clean venv: `pip install -e`
writes a `.pth` + import finder into `site-packages` whose only mapping is the
top-level `luna_hifi_tasks` -> this folder; everything below it resolves off the
filesystem at import time. So editing any `.py`, adding a new subpackage, or
adding an `__init__.py` to an existing folder all take effect on the next run
with no reinstall. Entry points and package discovery, by contrast, are baked
into `site-packages/*.dist-info` at install time, so a `pyproject.toml` change
is invisible until you reinstall.

**Missing `__init__.py`.** setuptools' `packages.find` only treats folders with
an `__init__.py` as packages, and `from . import agents` fails without one, so
imports silently register nothing. Adding the file is the whole fix; no
reinstall is required.

**`luna_hifi_tasks.egg-info/` is a build leftover, not live metadata.** The
metadata the CLI reads is `site-packages/luna_hifi_tasks-0.1.0.dist-info`.
Deleting the source-tree `egg-info` does not break task discovery (tested).
It is gitignored.

**MJCF cannot be imported at runtime.** `spawn_from_mjcf` calls
`isaacsim.asset.importer.mjcf`, a Kit extension that only exists after
`SimulationApp` boots. The CLI never boots Kit. Convert to USD offline.

**One solver per `NewtonCfg`.** Rigid + MPM run as two `CouplerEntryCfg`
entries inside a `CouplerProxyCfg`, assigned in `__post_init__`. Reference
implementation:
`source\isaaclab_tasks\isaaclab_tasks\contrib\ur10_particle_push\ur10_particle_push_env_cfg.py`.
Copy its structure.

- MPM entry MUST set `in_place=True` and `all_particles=True`
- `project_outside_colliders` MUST be `False` on a coupled entry
- tool bodies go in `CouplerProxyMappingCfg`, `mode="lagged"`, with a
  `mass_scale` (the lagged-feedback stability knob)
- `collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0)`

**MPM entries only see bodies you list.** The world ground plane is not among
them, so soil needs an explicit hidden kinematic slab (`MPMGround` in our cfg)
or particles fall forever.

**Sparse-grid capacities are absolute totals, not per-env.** Lowering
`--num_envs` does not shrink them. Must satisfy
`upper <= lower <= leaf <= active`. Current working values at 4 envs on 6 GB:
`1<<14 / 1<<13 / 1<<12 / 1<<10` (active/leaf/lower/upper). Hydra applies
`--num_envs` after `__post_init__`, so these do NOT auto-scale — raise them by
hand for more envs.

**Particle rendering is opt-in.** `NewtonGLVisualizerCfg.show_particles`
defaults to `False`. Set it in a `play_mode()` override on the env cfg.

**Prim path conventions.** Scene cfgs use `{ENV_REGEX_NS}`; coupler `bodies`
lists use the expanded `/World/envs/env_.*` form. The MJCF converter nests
bodies under `Geometry/chassis`, so wheel paths are
`/World/envs/env_.*/Tricycle/Geometry/chassis/.*_body`.

**`projected_gravity_b[:, 2]` reads +1.0 upright** for this converted asset,
not -1.0. The flip check is `< 0.3`, not `> -0.3`.

**`_setup_scene` must not construct assets.** `InteractiveScene` already built
everything in the scene cfg. Fetch by name: `self.car = self.scene["tricycle"]`.

**`configclass` import** is `from isaaclab.utils.configclass import configclass`.
Clear `__pycache__` after changing it — stale bytecode produced a phantom
"'module' object is not callable".

**Material fields.** `MPMParticleMaterialCfg` has: `density`, `young_modulus`,
`poisson_ratio`, `viscosity`, `friction`, `damping`, `yield_pressure`,
`tensile_yield_ratio`, `yield_stress`, `hardening`, `dilatancy`. No
`hardening_rate` or `softening_rate`.

**Soil parameter semantics** (differs from a hand-rolled MPM): the yield
surface is `tau_max(p) = yield_stress + friction * (p - p_min)`, so `friction`
≈ tan(phi) and `yield_stress` is cohesion in **Pa**. Do not port old
Drucker-Prager alpha or log-strain cohesion numbers directly.

---

## 4. Troubleshooting

| Symptom | Cause |
|---|---|
| `TypeError: Union[arg, ...]` on newton import | Python 3.10 |
| imgui-bundle builds from source and fails | Python 3.10 |
| `isaaclab.bat not recognized` | need `.\` prefix, and `cd` to IsaacLab |
| `No module named 'isaaclab_rl'` | venv not activated in this window |
| `NameNotFound: Environment ... doesn't exist` | package not installed, or `pyproject.toml` entry point edited without reinstalling |
| gym registry empty after import | missing `__init__.py` (just add it; no reinstall) |
| `A prim already exists at path` | `_setup_scene` constructing assets |
| capacity exceeded for upper/lower/leaf | raise that cap, keep ordering |
| `Failed to create volume` + CUDA 700 spam | OOM; caps too high. One failure, not hundreds |
| `Mean episode length: 1.00` | termination firing immediately |
| reward explodes to 1e8+ | physics diverged; needs invalid-state termination |
| soil invisible | `show_particles=False` default |

After any CUDA 700, check `nvidia-smi` and kill orphaned python processes
before retrying — the context is poisoned and VRAM may still be held.

---

## 5. Known-good tricycle config

- 4 envs, `env_spacing=2.0`
- strip 1.5 x 0.6 x 0.06 m, voxel 0.05, 1 particle/cell (~720 cells/env)
- `dt=1/100`, `decimation=2` (policy 50 Hz), `episode_length_s=2.0`
- actions: `[throttle, steer]`, obs 15, reward = clamped forward displacement
- `proxy_mass_scale=10.0`, `max_wheel_speed=8.0`
- ~90-160 steps/s

Ceiling: coupled MPM on a 6 GB laptop card is a handful of envs. For real
rover training either use a bigger GPU, or use Bekker/SCM terramechanics for
throughput and keep MPM for validation.

> The committed `tricycle_env_cfg.py` still carries the pre-tuning defaults
> `num_envs=8` and `max_wheel_speed=20.0`. The values above are what actually
> ran on the 6 GB card; `--num_envs 4` overrides the first from the CLI, the
> second has to be edited in the cfg.

**Body shape (2026-09-03).** The chassis is a triangular frame (two side rails,
a rear crossbar and a flat deck, all boxes) rather than a single box. Wheel
positions, radii, masses, joint names and body names are unchanged, so nothing
in the env cfg or the coupler mapping moved. The committed USD in
`assets\tricycle\` matches. If the car ever looks wrong after a pull, re-run
the "Convert the tricycle asset" step above; the converter output is the source
of truth and takes under a minute.

---

## 6. What this template covers, and what it doesn't

Covered — enough to start a new Newton task by copying `tricycle/`:

- package layout, `isaaclab.tasks` entry point, `gym.register`
- `DirectRLEnv`: `_setup_scene`, `_pre_physics_step`, `_apply_action`,
  `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`
- `SimulationCfg` / `NewtonCfg` / `MJWarpSolverCfg`
- `ArticulationCfg` with velocity- and position-drive actuators
- rigid ↔ implicit-MPM coupling via `CouplerProxyCfg` (the hard part)
- rsl_rl PPO runner cfg
- the offline MJCF → USD asset pipeline
- batched reset, including `MPMObject.reset(env_ids)`
- a `play_mode()` override that turns on particle rendering

Not covered — you'll be reading Isaac Lab's own tasks for these:

- **manager-based envs** (`RewardTermCfg`, `ObservationTermCfg`,
  `EventTermCfg`). This is Direct-style only, and a lot of Isaac Lab tasks
  use the manager style, which looks quite different.
- **sensors** — no contact sensors, cameras / tiled rendering, or ray-cast
  height scanners
- **domain randomisation, events, curriculum**
- **other RL libraries** — rsl_rl only; no skrl / rl_games / sb3 cfgs
- **terrain generation** — flat strip, no `TerrainImporterCfg`
- **reading MPM particle state into observations** (height maps, volume in a
  moving frame). Doable with warp kernels over `MPMObject.data`, but nothing
  here demonstrates it.
- **multi-term shaped rewards** — reward here is a single distance term
- **hierarchical / multi-agent RL**
