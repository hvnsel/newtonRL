# luna_hifi — Isaac Lab + Newton MPM tricycle task

An Isaac Lab task that runs an articulated vehicle and implicit-MPM soil in the
same simulation, trained with rsl_rl PPO.

A three-wheeled car drives forward across a strip of MPM soil. Actions are
`[throttle, steer]`, the observation is 15-dimensional, and reward is forward
displacement per step, which sums to metres travelled over the 2 s episode.

Built against the Isaac Lab **develop** branch (3.0 line) with the Newton
backend. API reference:
<https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/api/index.html>

`docs/tricycle_manual/main.pdf` explains every file line by line.

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
  assets/tricycle/tricycle.usda     converted car asset
  docs/tricycle_manual/             the line-by-line manual (LaTeX source + PDF)
```

---

## Requirements

- Windows 11
- Python 3.12
- An NVIDIA GPU. The known-good configuration below runs on a 6 GB card.

Paths below use `C:\dev` as the workspace root. Substitute your own; Isaac Lab
and this repo sit side by side:

```
C:\dev\
    IsaacLab\                       framework, develop branch, kit-less + isaacsim pip pkg
        .venv\                      Python 3.12 venv — everything runs through this
    luna_hifi\                      THIS REPO
```

---

## 1. One-time setup

### Python 3.12

```powershell
winget install --id Python.Python.3.12 -e
# reopen PowerShell
py -0                                   # -V:3.12 appears in the list
```

### Isaac Lab

```powershell
cd C:\dev
git clone -b develop https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
.\isaaclab.bat -i
.\isaaclab.bat -i isaacsim        # the MJCF->USD converter needs this
```

Confirm the framework runs on its own:

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 physics=newton_mjwarp --viz newton
```

### Install this package

```powershell
.\isaaclab.bat -p -m pip install -e C:\dev\luna_hifi
.\isaaclab.bat -p -c "import luna_hifi_tasks, gymnasium as gym; print([k for k in gym.registry if 'Luna-' in k])"
```

The second command prints `['Luna-Tricycle-Direct']`.

### Convert the tricycle asset

`tricycle.py` is the source of truth for the car. `write_mjcf()` emits it as
MJCF, and Isaac Lab's converter turns that into the USD the task loads. Run
this once, and again whenever `tricycle.py` changes:

```powershell
.\isaaclab.bat -p -c "from luna_hifi_tasks.tricycle.tricycle import write_mjcf; print(write_mjcf(r'C:\dev\luna_hifi\assets'))"
.\isaaclab.bat -p scripts\tools\convert_mjcf.py C:\dev\luna_hifi\assets\tricycle.xml C:\dev\luna_hifi\assets\tricycle.usd
```

The converter writes `assets\tricycle\tricycle.usda`. `tricycle_env_cfg.py`
resolves that path relative to the package, so the repo can live anywhere.
Conversion takes under a minute.

---

## 2. Daily commands

Every session, in each new PowerShell window:

```powershell
cd C:\dev\IsaacLab
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Train:

```powershell
.\isaaclab.bat train --rl_library rsl_rl --task Luna-Tricycle-Direct --num_envs 4
```

Play a trained checkpoint, with soil particles visible:

```powershell
.\isaaclab.bat play --rl_library rsl_rl --task Luna-Tricycle-Direct --num_envs 1 --checkpoint latest --viz newton
```

Add `--viz newton` to watch training live. Logs and checkpoints land in
`logs\rsl_rl\tricycle\<timestamp>\`, saved every 50 iterations.

---

## 3. How it fits together

**CLI.** The entry points are `isaaclab train|play|zero_agent|random_agent`.
Headless is the default. The visualiser flag is `--viz`. Task IDs carry no
version suffix. The physics backend is selected with the Hydra config group
`physics=newton_mjwarp`.

**Task discovery.** Isaac Lab finds external tasks through the
`isaaclab.tasks` entry point in `pyproject.toml`. Entry points and package
discovery are baked into `site-packages/*.dist-info` at install time, so
reinstall with `-e` after editing `pyproject.toml`. `pip install -e` maps the
top-level `luna_hifi_tasks` to this folder and resolves everything below it off
the filesystem at import time, so `.py` edits and new subpackages take effect
on the next run.

**Assets are converted offline.** `spawn_from_mjcf` calls
`isaacsim.asset.importer.mjcf`, a Kit extension available only after
`SimulationApp` boots, and the CLI runs Kit-less. The car therefore ships as
USD, converted by the step above.

**One solver per `NewtonCfg`.** Rigid bodies and MPM run as two
`CouplerEntryCfg` entries inside a `CouplerProxyCfg`, assigned in
`__post_init__`. Reference implementation:
`source\isaaclab_tasks\isaaclab_tasks\contrib\ur10_particle_push\ur10_particle_push_env_cfg.py`.

- the MPM entry sets `in_place=True` and `all_particles=True`
- a coupled entry sets `project_outside_colliders=False`
- tool bodies go in `CouplerProxyMappingCfg`, `mode="lagged"`, with a
  `mass_scale` — the lagged-feedback stability knob
- `collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0)`

**MPM entries see only the bodies listed on them.** The soil rests on an
explicit hidden kinematic slab, `MPMGround` in the cfg.

**Sparse-grid capacities are absolute totals**, independent of `--num_envs`,
and must satisfy `upper <= lower <= leaf <= active`. The committed values are
`1<<14 / 1<<13 / 1<<12 / 1<<10` (active / leaf / lower / upper), sized for 4
envs on 6 GB. Hydra applies `--num_envs` after `__post_init__`, so raise these
by hand for more envs.

**Particle rendering** is enabled by `NewtonGLVisualizerCfg.show_particles` in
the cfg's `play_mode()` override.

**Prim paths.** Scene cfgs use `{ENV_REGEX_NS}`; coupler `bodies` lists use the
expanded `/World/envs/env_.*` form. The MJCF converter nests bodies under
`Geometry/chassis`, so wheel paths are
`/World/envs/env_.*/Tricycle/Geometry/chassis/.*_body`.

**Env construction.** `InteractiveScene` builds everything in the scene cfg, so
`_setup_scene` fetches assets by name: `self.car = self.scene["tricycle"]`.

**Orientation.** `projected_gravity_b[:, 2]` reads +1.0 upright for this
converted asset, and the flip check is `< 0.3`.

**Soil material.** `MPMParticleMaterialCfg` takes `density`, `young_modulus`,
`poisson_ratio`, `viscosity`, `friction`, `damping`, `yield_pressure`,
`tensile_yield_ratio`, `yield_stress`, `hardening`, `dilatancy`. The yield
surface is `tau_max(p) = yield_stress + friction * (p - p_min)`, so `friction`
is tan(phi) and `yield_stress` is cohesion in **Pa**.

---

## 4. Configuration

Shipped in `tricycle_env_cfg.py`:

- `num_envs=8`, `env_spacing=2.0`
- soil strip 1.5 x 0.6 x 0.06 m, voxel 0.05, 1 particle/cell (~720 cells/env)
- `dt=1/100`, `decimation=2` (policy at 50 Hz), `episode_length_s=2.0`
- `max_wheel_speed=20.0`, `proxy_mass_scale=10.0`
- `play_mode()` clamps to 4 envs and turns on particle rendering

Validated on a 6 GB card at `--num_envs 4` with `max_wheel_speed` lowered to
`8.0` in the cfg, running 90-160 steps/s. Coupled MPM on a card that size fits
a handful of envs; larger runs want a bigger GPU, or Bekker/SCM terramechanics
for throughput with MPM kept for validation.

The chassis is a triangular frame — two side rails, a rear crossbar and a flat
deck, all boxes. Wheel positions, radii, masses, joint names and body names are
independent of it, so the env cfg and coupler mapping are unaffected by chassis
edits.

---

## 5. What this template covers

Enough to start a new Newton task by copying `tricycle/`:

- package layout, `isaaclab.tasks` entry point, `gym.register`
- `DirectRLEnv`: `_setup_scene`, `_pre_physics_step`, `_apply_action`,
  `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`
- `SimulationCfg` / `NewtonCfg` / `MJWarpSolverCfg`
- `ArticulationCfg` with velocity- and position-drive actuators
- rigid ↔ implicit-MPM coupling via `CouplerProxyCfg`
- rsl_rl PPO runner cfg
- the offline MJCF → USD asset pipeline
- batched reset, including `MPMObject.reset(env_ids)`
- a `play_mode()` override for particle rendering

The task is Direct-style with rsl_rl. Manager-based envs, sensors, domain
randomisation, terrain generation and other RL libraries are covered by Isaac
Lab's own task suite.
