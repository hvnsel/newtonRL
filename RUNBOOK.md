# Luna_HiFi — Isaac Lab + Newton MPM Runbook

Working setup as of 2026-09-03, Windows 11, RTX A1000 6 GB laptop.
Validated end to end: the tricycle task trains and plays with soil particles visible.

---

## 0. What's where

```
C:\Users\hanse\Documents\
    IsaacLab\                       framework, develop branch, kit-less + isaacsim pip pkg
        .venv\                      Python 3.12 venv — everything runs through this
    Luna_HiFi\                      OUR CODE
        pyproject.toml              declares the isaaclab.tasks entry point
        luna_hifi_tasks\
            __init__.py             imports subpackages -> triggers gym.register
            tricycle\               pipeline validation task (works)
            excavation\             rover task (template, not yet run)
        assets\tricycle\tricycle.usda   converted car asset
        newton_soil.py              standalone Newton module (NOT used by Isaac Lab)
        free_soil.py, sphere_*.py, bowl_scoop_demo.py   standalone demos
        TricycleDemo\               generated scaffold — NOT NEEDED, safe to delete
    isaacsim\                       old Isaac Sim install, unrelated, leave alone
```

Standalone demos vs Isaac Lab: the demos are for soil calibration and
visualisation only. All training happens in Isaac Lab.

---

## 1. One-time setup

### Python 3.12 (3.10 does NOT work)

Newton 1.5 annotates with `wp.array[wp.bool] | None`. Python 3.10's
`typing._type_check` rejects that; 3.11+ doesn't. Also imgui-bundle's
newest cp310 Windows wheel is 1.5.2, far below newton's `>=1.92.0` floor,
so pip falls back to a source build that fails.

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

A pyglet `AssertionError` after "Training time:" is a harmless
window-teardown race. Ignore it.

### Install our package

```powershell
.\isaaclab.bat -p -m pip install -e C:\Users\hanse\Documents\Luna_HiFi
.\isaaclab.bat -p -c "import luna_hifi_tasks, gymnasium as gym; print([k for k in gym.registry if 'Luna-' in k])"
```

Must print `['Luna-Tricycle-Direct']`. Empty list = registration broken,
see Troubleshooting.

### Convert the tricycle asset (once)

```powershell
mkdir C:\Users\hanse\Documents\Luna_HiFi\assets
.\isaaclab.bat -p -c "from luna_hifi_tasks.tricycle.tricycle import write_mjcf; print(write_mjcf())"
.\isaaclab.bat -p scripts\tools\convert_mjcf.py C:\Users\hanse\AppData\Local\Temp\luna_hifi_assets\tricycle.xml C:\Users\hanse\Documents\Luna_HiFi\assets\tricycle.usd
```

Note: it writes `assets\tricycle\tricycle.usda` (subfolder, `.usda`), not the
path you gave. `tricycle_env_cfg.py` points at the real output.
Redo this whenever `tricycle.py` changes.

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
No `--headless` flag; headless is the default. `--viz`, not
`--visualizer`. Task IDs carry no `-v0` suffix. Backend override is
`physics=newton_mjwarp` (a Hydra config group), not `presets=newton`.

**External task discovery** is via the `isaaclab.tasks` entry point in
`pyproject.toml`. Entry points bake in at install time — after editing
them, reinstall with `-e`. The generated `TricycleDemo` scaffold is not
needed; entry points make it redundant.

**Editable install + missing `__init__.py`.** setuptools' `packages.find`
skips directories without `__init__.py`, so the package installs as an
empty namespace and imports silently register nothing. If a directory
gains an `__init__.py` after install, reinstall.

**MJCF cannot be imported at runtime.** `spawn_from_mjcf` calls
`isaacsim.asset.importer.mjcf`, a Kit extension that only exists after
`SimulationApp` boots. The CLI never boots Kit. Convert to USD offline.

**One solver per `NewtonCfg`.** Rigid + MPM run as two `CouplerEntryCfg`
entries inside a `CouplerProxyCfg`, assigned in `__post_init__`. Reference
implementation: `source\isaaclab_tasks\isaaclab_tasks\contrib\ur10_particle_push\ur10_particle_push_env_cfg.py`.
Copy its structure.

- MPM entry MUST set `in_place=True` and `all_particles=True`
- `project_outside_colliders` MUST be `False` on a coupled entry
- tool bodies go in `CouplerProxyMappingCfg`, `mode="lagged"`, with a
  `mass_scale` (the lagged-feedback stability knob)
- `collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0)`

**MPM entries only see bodies you list.** The world ground plane is not
among them, so soil needs an explicit hidden kinematic slab
(`MPMGround` in our cfg) or particles fall forever.

**Sparse-grid capacities are absolute totals, not per-env.** Lowering
`--num_envs` does not shrink them. Must satisfy
`upper <= lower <= leaf <= active`. Current working values at 4 envs on
6 GB: `1<<14 / 1<<13 / 1<<12 / 1<<10` (active/leaf/lower/upper). Hydra
applies `--num_envs` after `__post_init__`, so these do NOT auto-scale —
raise them by hand for more envs.

**Particle rendering is opt-in.** `NewtonGLVisualizerCfg.show_particles`
defaults to `False`. Set it in a `play_mode()` override on the env cfg.

**Prim path conventions.** Scene cfgs use `{ENV_REGEX_NS}`; coupler
`bodies` lists use the expanded `/World/envs/env_.*` form. The MJCF
converter nests bodies under `Geometry/chassis`, so wheel paths are
`/World/envs/env_.*/Tricycle/Geometry/chassis/.*_body`.

**`projected_gravity_b[:, 2]` reads +1.0 upright** for this converted
asset, not -1.0. The flip check is `< 0.3`, not `> -0.3`.

**`_setup_scene` must not construct assets.** `InteractiveScene` already
built everything in the scene cfg. Fetch by name:
`self.car = self.scene["tricycle"]`.

**`configclass` import** is `from isaaclab.utils.configclass import configclass`.
Clear `__pycache__` after changing it — stale bytecode
produced a phantom "'module' object is not callable".

**Material fields.** `MPMParticleMaterialCfg` has: `density`, `young_modulus`,
`poisson_ratio`, `viscosity`, `friction`, `damping`, `yield_pressure`,
`tensile_yield_ratio`, `yield_stress`, `hardening`, `dilatancy`. No
`hardening_rate` or `softening_rate` — drop those from `SoilMaterial`
before calling `to_material_cfg()`.

**Soil parameter semantics** (differs from the old custom MPM):
the yield surface is `tau_max(p) = yield_stress + friction * (p - p_min)`, so
`friction` ≈ tan(phi) and `yield_stress` is cohesion in **Pa**. Do not
port old Drucker-Prager alpha or log-strain cohesion numbers directly.

---

## 4. Troubleshooting

| Symptom | Cause |
|---|---|
| `TypeError: Union[arg, ...]` on newton import | Python 3.10 |
| imgui-bundle builds from source and fails | Python 3.10 |
| `isaaclab.bat not recognized` | need `.\` prefix, and `cd` to IsaacLab |
| `No module named 'isaaclab_rl'` | venv not activated in this window |
| `NameNotFound: Environment ... doesn't exist` | entry point missing, or reinstall needed |
| gym registry empty after import | missing `__init__.py`, then reinstall |
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
rover training either use a bigger GPU, or use Bekker/SCM terramechanics
for throughput and keep MPM for validation.

> The committed `tricycle_env_cfg.py` still carries the pre-tuning defaults
> `num_envs=8` and `max_wheel_speed=20.0`. The values above are what actually
> ran on the 6 GB card; `--num_envs 4` overrides the first from the CLI, the
> second has to be edited in the cfg.

**Body shape (2026-09-03).** The chassis is now a triangular frame (two side
rails, a rear crossbar and a flat deck, all boxes) instead of the single box.
Wheel positions, radii, masses, joint names and body names are unchanged, so
nothing in the env cfg or the coupler mapping moved. The committed USD in
`assets\tricycle\` was regenerated to match. If the car ever looks wrong after
a pull, re-run the "Convert the tricycle asset" step above; the converter
output is the source of truth and takes under a minute.

---

## 6. Next steps

1. Tricycle: confirm reward climbs past ~2 m without diverging.
2. Excavation task (`luna_hifi_tasks/excavation/`) still has STUB and
   VERIFY markers. Its `NewtonCfg` block needs the same `CouplerProxyCfg`
   rewrite the tricycle got, plus `_setup_scene`, `configclass` import,
   and `play_mode()` fixes.
3. Drop in the real rover USD, fix the joint regexes and wheel prim paths.
4. Implement the height-map and bucket-fill kernels in
   `excavation/soil.py` — currently unvalidated stubs.
