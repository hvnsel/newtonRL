# Where the knobs are

| file | holds |
|---|---|
| `luna_hifi_tasks/excavator/excavator.py` | the machine. A change to the MJCF needs the asset rebuilt (README.md, "Assets"). |
| `.../excavate/excavate_env_cfg.py` | excavate: soil, bed, solver, rewards |
| `.../navigate/navigate_env_cfg.py` | navigate: terrain, goals, planner, curriculum, rewards |
| `.../*/agents/rsl_rl_ppo_cfg.py` | PPO: rollout length, discount, networks |
| `.../excavator_cfg.py` | actuators and speed limits |
| `.../mdp/observations.py` | observation layouts and scan geometry. A changed layout does not load old checkpoints. |

`scripts/dig_demo.py` takes `--cohesion`, `--friction`, `--density`, `--voxel`,
`--cut`, `--drum` and `--drive` as flags.

## Soil (excavate_env_cfg.py)

| what | field | value | note |
|---|---|---|---|
| bed height-grid cell | `voxel_size` | `0.04` m | scans and cut depth are quantised to it |
| MPM grid cell | `mpm_voxel_size` | `0.08` m | `--voxel` |
| particles per cell, per axis | `particles_per_cell` | `2.0` | 8 per cell |
| density | `soil_density` | `1500` kg/m³ | `--density` |
| friction | `soil_friction` | `0.84` | about tan(phi); `--friction` |
| cohesion | `soil_cohesion` | `0` Pa (Small/Micro `800`) | `yield_stress`; `--cohesion` |
| compression cap | `soil_yield_pressure` | `1e12` Pa | |

A soil field changed after `__post_init__` takes effect through
`cfg.apply_soil_material()`.

## Bed

| what | field | Excavate | Small | Micro |
|---|---|---|---|---|
| x extent | `bed_x` | `(-1.92, 2.64)` | `(1.80, 2.70)` | `(1.80, 2.80)` |
| y extent | `bed_y` | `(-0.80, 0.80)` | `(-0.55, 0.55)` | `(-0.50, 0.50)` |
| depth | `bed_depth` | `0.25` | `0.16` | `0.21` |
| machine spawns on the bed | `spawn_on_bed` | `True` | `False` | `False` |
| episodes between soil resets | `soil_reset_every` | `1` | `1` | `1` |
| envs the MPM grid is sized for | `max_num_envs` | `4` | `1` | `1` |

dig_demo's crawl speed is `--drive` × `MAX_WHEEL_SPEED` × `WHEEL_RADIUS`:
`--drive 0.05` is 0.075 m/s.

## Training

| what | field | excavate | navigate |
|---|---|---|---|
| physics rate | `sim.dt` | 100 Hz | 200 Hz |
| policy rate | `decimation` | `4` (25 Hz) | `8` (25 Hz) |
| episode | `episode_length_s` | `16` s (Small/Micro `30`) | `30` s |
| unpaid settle at the start | `settle_seconds` | `1` s | |
| discount | `gamma` | `0.997` | `0.997` |
| rollout | `num_steps_per_env` | `256` | `128` |
| envs | `scene.num_envs` | `16` | `256` |
| checkpoint every | `save_interval` | `5` | `25` |

The value horizon is `1 / (1 - gamma)` policy steps: 333 steps, 13 s at 25 Hz.

Reward weights are the `w_<term>` fields of each env cfg; `Episode_Reward/<term>`
in the training log is each term's episode total.

| excavate | field | value |
|---|---|---|
| full drum | `target_load_kg` | `40` kg |
| what counts as in a drum | `BORE_*` constants | inside the shroud shell: within 0.195 m of the axis, between the end plates, above the chord across the inlet; holds 150 kg |
| nominal cut volume | `cut_volume_ref` | `0.0304` m³ |
| commanded depth below the surface | `cut_depth_range` | `(0.04, 0.04)` m |
| drum full (logged as `Episode/drum_full`) | `fill_success_fraction` | `0.6` of `target_load_kg`, front drum |

## Excavate tool and drive (excavate_env_cfg.py)

The env runs the tool; the policy's actions are `[forward, yaw]`.

| what | field | value |
|---|---|---|
| shroud bottom below the surface | `dig_cut` | `0.10` m (`boom_command_for_cut`: arm 0.536 rad) |
| inlet off straight down | `dig_inlet_tilt` | `0.096` rad (shroud −0.44 rad) |
| drum speed | `dig_drum_speed` | `−2.9` rad/s, the loading direction |
| lowering and spin-up after the settle | `dig_ramp_seconds` | `2` s, wheels held |
| forward | `dig_forward` | `0.15 × (1 + a0)` of top speed: 0–0.45 m/s, 0.225 at a0 = 0 |
| heading hold, yaw action | `heading_hold_gain`, `dig_yaw_rate` | 1.0 /s capped at ±0.3 rad/s, plus 0.3 × a1 rad/s |
| end of the pass | `work_area_length` | the episode ends when the front drum is 0.6 m past the anchor |
| rear drum | `dig_rear_drum` | `False`: raised at the start pose, still |
| tool off (parked machine) | `dig_tool` | `True` |

## Navigate planner (navigate_env_cfg.py)

| what | field | value |
|---|---|---|
| grid | `path_res` | `0.25` m over the 32 m cell |
| impassable band round an obstacle | `path_clearance` | `0.70` m (machine half-width 0.675) |
| cost zone beyond it | `path_zone`, `path_zone_cost` | 2.0 m, from 3 down to 1 per metre |
| inside the band | `path_obstacle_cost` | `50` per metre |
| path points in the observation | `path_lookahead` | 1 m and 2 m ahead, body frame |

## Sensing

| what | where | value |
|---|---|---|
| navigate far scan (critic) | `NAV_FAR_*` | 27 × 27 at 0.45 m, centred: 5.85 m each way |
| navigate near scan (critic, collision check) | `NAV_NEAR_*` | 25 × 25 at 0.20 m, centred: 2.4 m each way |
| excavate scan | `DIG_SCAN_*` | 16 × 8 at 0.125 m, centred on the front drum |
| excavate scan noise, dropout | `scan_noise_std`, `scan_dropout` | `0.015` / `0.02`, growing with range; a dropped cell reads `scan_invalid` |
| actor's drum load sensor | `fill_sensor_*` | 0.3 s lag, 5% noise; the critic reads the exact mass |

Both navigate scans are centred, so the 180-degree symmetry (mdp/symmetry.py)
maps each onto itself.

## Rotor (excavator.py)

| what | constant | value |
|---|---|---|
| vanes | `ROTOR_VANES` | `4` |
| rake | `ROTOR_RAKE` | 32° |
| hub, tip radius | `ROTOR_HUB_R`, `ROTOR_TIP_R` | `0.045`, `0.185` m |
| half-length | `ROTOR_HALF_LEN` | `0.475` m |
| vane half-thickness | `VANE_HALF_T` | `0.006` m |
| boxes per vane | `VANE_SEGMENTS` | `5` |
| spiral handedness | `RAKE_SIGN` | `1.0` |

A vane sweeps `ln(TIP/HUB) · tan(rake)` = 50.6° from hub to tip, inside its
90° pocket. The chord between adjacent vane tips, `VANE_TIP_GAP`, is 0.262 m.

## Shroud

| what | constant | value |
|---|---|---|
| inlet | `SHROUD_INLET` | 100° |
| inner, outer radius | `SHROUD_IN_R`, `SHROUD_OUT_R` | `0.195`, `0.212` m |
| end plate thickness | `SHROUD_END_T` | `0.012` m |
| aiming range | `SHROUD_RANGE` | `(-1.10, 1.10)` rad |

`SHROUD_IN_R - ROTOR_TIP_R` is the 10 mm running clearance.

## Machine

| what | constant | value |
|---|---|---|
| wheelbase, track | `WHEELBASE`, `TRACK` | `1.40`, `1.15` m |
| wheel radius, half-width | `WHEEL_RADIUS`, `WHEEL_HALF_W` | `0.30`, `0.10` m |
| grousers per wheel | `WHEEL_GROUSERS` | `6` |
| arm length | `ARM_LEN` | `0.76` m |
| arm pivot ahead of the axle | `MAST_OFFSET_X` | `0.22` m |
| arm travel | `ARM_RANGE` | `(-0.55, 0.72)` rad; 0.72 buries the vane tips 0.185 m |
| ballast deck | `DECK_MASS` | `670` kg; machine total 994 kg |

## Rates and the solver

| what | where | value |
|---|---|---|
| wheel speed cap | `MAX_WHEEL_SPEED` (excavator_cfg.py) | `5.0` rad/s (1.5 m/s) |
| drum speed cap | `MAX_DRUM_SPEED` | `2.9` rad/s |
| contact buffers | `rigid_njmax`, `rigid_nconmax` | `8192`, `32768` |
| MPM sweeps per solve | `mpm_max_iterations` | `50` (a multiple of 25) |
| grid cells per particle | `grid_cap_multiplier` | `1.0` |
| leaf size | `grid_leaf_shift` | `3` |

`SPIN_LIMIT = sqrt(g / ROTOR_TIP_R)` = 2.96 rad/s is the drum speed at which
the vane tips' outward acceleration equals lunar gravity. An overrun contact
buffer shows up as CUDA error 700.
