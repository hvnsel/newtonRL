# luna_hifi — lunar regolith excavator RL on Isaac Lab + Newton

Isaac Lab 3.0 tasks (kit-less, Newton backend) trained with rsl_rl PPO, for a
four-wheel skid-steer excavator with a counter-rotating bucket drum at each
end.

| task id | what |
|---|---|
| `Luna-Excavator-Navigate` | drive to a goal pose over generated terrain with craters and boulders (MJWarp, rigid terrain mesh) |
| `Luna-Excavator-Navigate-Course` | the navigate policy on one fixed obstacle layout (mdp/course.py), for playback |
| `Luna-Excavator-Excavate` | cut to a target depth in an implicit-MPM soil bed and carry the spoil |
| `Luna-Excavator-Excavate-Small`, `-Micro` | one env, a small soil pile in front of the machine |
| `Luna-Tricycle-Direct` | a tricycle on an MPM strip: the minimal rigid ↔ MPM coupling example |

`docs/tricycle_manual/main.pdf` explains the tricycle task line by line.

## Layout

```
pyproject.toml                    the isaaclab.tasks entry point
luna_hifi_tasks/
  __init__.py                     imports the subpackages, which gym.register the tasks
  ground.py                       ground plane from assets/ground_plane/
  excavator/
    excavator.py                  the machine as an MJCF string; write_mjcf()
    excavator_cfg.py              ArticulationCfg, actuators, prim paths, speed limits, usd_status()
    excavator_env_base.py         DirectRLEnv base shared by navigate and excavate
    terrain_cfg.py                navigate's terrain generator and the course cell
    mdp/
      observations.py             observation layouts and scan geometry
      rewards.py                  shared reward terms and the per-term episode logger
      terrain.py                  scan patterns, height sampling, terrain height fields
      paths.py                    path-cost field to the goal (progress and bearing)
      course.py                   fixed obstacle layouts for Navigate-Course
      symmetry.py                 180-degree symmetry for PPO augmentation
      sensors.py                  MPM soil readouts for excavate
    navigate/                     navigate env, cfg, PPO cfg
    excavate/                     excavate env, cfg, PPO cfg
  tricycle/                       tricycle env, cfg, MJCF, PPO cfg
  planner/site.py                 heightfield site model with greedy and beam dig planners
assets/                           converted USDs (excavator USD is built per machine, see below)
patches/                          Newton implicit-MPM patch
scripts/
  peek.py                         print a run's TensorBoard scalars
  read_trace.py, plot_nav.py      navigate playback traces: tables and figures
  dig_demo.py                     scripted dig on the excavate task, no policy
  play_dig.py                     excavate playback with a CSV trace and heightmap dump
  render_dig.py, render_attitude.py   GIFs from play_dig dumps
  patch_newton_mpm.py             patches the installed Newton MPM solver
workstation/                      single-box setup and run scripts (Setup.md)
cluster/                          Georgia Tech PACE setup (ClusterSetup.md)
```

## Setup

- Workstation: `workstation/Setup.md`
- PACE: `cluster/ClusterSetup.md`

## Assets

Training loads `assets/excavator/excavator.usda`, converted from the MJCF in
`excavator.py`; `assets/excavator.usd.srchash` records the MJCF hash it was
built from. Build both once per machine and after any change to the MJCF. The
converter needs Isaac Sim in the venv (`uv sync --extra isaacsim`).

```bash
cd ~/newtonRL
~/IsaacLab/.venv/bin/python -c "from luna_hifi_tasks.excavator.excavator import write_mjcf; print(write_mjcf())"
rm -rf assets/excavator
~/IsaacLab/.venv/bin/python ~/IsaacLab/scripts/tools/convert_mjcf.py <printed path> assets/excavator.usd
```

The converter writes `assets/excavator/excavator.usda` from the output path
`assets/excavator.usd`. The excavate env stops with these commands when the
stamp does not match.

The tricycle asset is checked in; after editing `tricycle.py`:

```bash
~/IsaacLab/.venv/bin/python -c "from luna_hifi_tasks.tricycle.tricycle import write_mjcf; print(write_mjcf())"
~/IsaacLab/.venv/bin/python ~/IsaacLab/scripts/tools/convert_mjcf.py <printed path> assets/tricycle.usd
```

## Commands

From the repo root (`workstation/train.sh` runs the Isaac Lab CLI; logs go to
`logs/rsl_rl/<experiment>/<timestamp>/`):

```bash
# navigate
workstation/train.sh train --rl_library rsl_rl --task Luna-Excavator-Navigate \
  --num_envs 256 > train.log 2>&1 &
# resume: the curriculum is not in the checkpoint, so pass the goal range and
# terrain row reached (Curriculum/goal_dist_hi, Curriculum/terrain_level)
workstation/train.sh train --rl_library rsl_rl --task Luna-Excavator-Navigate \
  --num_envs 256 --checkpoint logs/rsl_rl/excavator_navigate/<run>/model_<N>.pt \
  --max_iterations <remaining> "env.goal_dist_range=[1.5,<goal_dist_hi>]" \
  env.scene.terrain.max_init_terrain_level=<row> > train.log 2>&1 &
python scripts/peek.py --tags goal_dist_hi reached mean_std learning_rate
bash workstation/nav_videos.sh          # clips, traces and figures in ~/nav_media
bash workstation/nav_course.sh          # the obstacle course in ~/nav_course

# excavate
workstation/train.sh train --rl_library rsl_rl --task Luna-Excavator-Excavate --num_envs 4
workstation/train.sh -p scripts/dig_demo.py --task Luna-Excavator-Excavate-Micro --no_window
workstation/train.sh -p scripts/play_dig.py --no_window --trace dig.csv

# tricycle
workstation/train.sh train --rl_library rsl_rl --task Luna-Tricycle-Direct --num_envs 4

# stop a run
pkill -f Luna-Excavator-Navigate
```

## Isaac Lab 3.0 / Newton notes

- **CLI.** `isaaclab train|play|zero_agent|random_agent`; headless by default;
  `--viz newton` opens the viewer. Task ids carry no `-v0`. The physics backend
  override is `physics=newton_mjwarp`.
- **Task discovery** is the `isaaclab.tasks` entry point in `pyproject.toml`.
  The editable install maps `luna_hifi_tasks` to this folder, so `.py` edits
  and new subpackages take effect on the next run; a `pyproject.toml` change
  needs `pip install -e .` again.
- **Every package folder needs an `__init__.py`**, or its `gym.register` calls
  never run.
- **MJCF is converted to USD offline.** `spawn_from_mjcf` needs a Kit
  extension the kit-less CLI does not load.
- **Rigid + MPM** run as two `CouplerEntryCfg` entries in a `CouplerProxyCfg`
  (reference: Isaac Lab's `contrib/ur10_particle_push`). The MPM entry sets
  `in_place=True` and `all_particles=True`, and `project_outside_colliders=False`;
  tool bodies go in `CouplerProxyMappingCfg` with `mode="lagged"` and a
  `mass_scale`; `collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0)`.
- **MPM sees only the bodies listed for it.** The soil rests on its own hidden
  kinematic slab, not on the world ground plane.
- **Sparse-grid capacities are totals over all envs**, set in `__post_init__`
  (before Hydra applies `--num_envs`), with `upper <= lower <= leaf <= active`.
- **Particles render only with** `NewtonGLVisualizerCfg.show_particles=True`.
- **Prim paths.** Scene cfgs use `{ENV_REGEX_NS}`; coupler body lists use
  `/World/envs/env_.*`. The converter nests bodies under `Geometry/<root body>`.
- **`projected_gravity_b[:, 2]` reads +1.0 upright** for these converted assets.
- **`_setup_scene` fetches assets** (`self.scene["name"]`); `InteractiveScene`
  builds everything in the scene cfg.
- **`configclass`** is `from isaaclab.utils.configclass import configclass`.
- **`MPMParticleMaterialCfg` fields:** `density`, `young_modulus`,
  `poisson_ratio`, `viscosity`, `friction`, `damping`, `yield_pressure`,
  `tensile_yield_ratio`, `yield_stress`, `hardening`, `dilatancy`. The yield
  surface is `tau_max(p) = yield_stress + friction * (p - p_min)`: `friction`
  is about tan(phi) and `yield_stress` is cohesion in Pa.

## Troubleshooting

| symptom | cause |
|---|---|
| `TypeError: Union[arg, ...]` on newton import | Python 3.10; Isaac Lab 3.0 needs 3.12 |
| `No module named 'isaaclab_rl'` | not the Isaac Lab venv's python |
| `NameNotFound: Environment ... doesn't exist` | package not installed, or entry point edited without reinstalling |
| gym registry empty after import | a missing `__init__.py` |
| `A prim already exists at path` | `_setup_scene` constructing assets |
| capacity exceeded for upper/lower/leaf | raise that cap, keeping the ordering |
| `Failed to create volume` + CUDA 700 | GPU out of memory: grid caps too high |
| `Mean episode length: 1.00` | a termination firing on the first step |
| reward explodes to 1e8+ | physics diverged |
| soil invisible | `show_particles=False` |

After a CUDA 700, kill leftover python processes (`nvidia-smi`) before
retrying.
