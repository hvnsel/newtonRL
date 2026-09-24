# Where the knobs are

Two files, and which one you touch decides whether you have to rebuild the USD.

| | |
|---|---|
| **`luna_hifi_tasks/excavator/excavator.py`** | the machine itself. **Changing anything here means re-running the converter** before the change reaches a run. |
| **`luna_hifi_tasks/excavator/excavate/excavate_env_cfg.py`** | the scene: soil, bed, solver. Takes effect on the next run. |

`scripts/dig_demo.py` exposes the most-swept scene knobs as flags so you do not
have to edit anything to try a value.

---

## Soil and particles

Particle size and particle spacing are the same number: the MPM voxel. Particles
are spawned one voxel apart and the coupler inflates every collider by half a
voxel per side, so the voxel sets both how big a grain is and how much it eats
out of every passage.

| what | where | now | note |
|---|---|---|---|
| voxel / grain size | `voxel_size` | `0.03` | `--voxel` on dig_demo. Halving it multiplies particle count by 8 |
| particles per cell | `particles_per_cell` | `1.0` | 2.0 quadruples the count for the same bed |
| density | `soil_density` | `1800` | `--density` |
| friction | `soil_friction` | `0.84` | ~tan(phi). `--friction` |
| cohesion | `soil_cohesion` | `800` | yield_stress in Pa. `--cohesion`. 0 sprays, ~800 holds a cut together |
| compaction cap | `soil_yield_pressure` | `1e12` | packed ground, not a sandbox |

All in `excavate_env_cfg.py`. A soil field set after `__post_init__` must be
followed by `cfg.apply_soil_material()`, which is why these are explicit flags
rather than Hydra overrides.

## The bed

| what | where | Small | Micro |
|---|---|---|---|
| extent along x | `bed_x` | `(1.80, 2.70)` | `(1.80, 2.80)` |
| extent along y | `bed_y` | `(-0.55, 0.55)` | `(-0.50, 0.50)` |
| depth | `bed_depth` | `0.16` | `0.21` |
| on the bed or beside it | `spawn_on_bed` | `False` | `False` |

The near edge is not arbitrary. At the dig angle the drum axis sits at
`PIVOT_X + ARM_LEN*cos(arm)` = 1.65 m with its leading edge at 1.86, so a bed
starting much before 1.80 puts the machine most of the way across it before it
has begun. dig_demo prints the drum's position against the bed and how long the
crawl takes to clear it.

Crawl speed is `--drive` x `MAX_WHEEL_SPEED` x `WHEEL_RADIUS`, so `--drive 0.05`
is 0.075 m/s and crosses a 0.9 m bed in twelve seconds. Anything above about
0.1 leaves the soil before the drum has filled.

## The rotor

All in `excavator.py`. **Rebuild the USD after changing any of these.**

| what | constant | now |
|---|---|---|
| vanes | `ROTOR_VANES` | `6` |
| rake | `ROTOR_RAKE` | `radians(32)` |
| hub and tip radius | `ROTOR_HUB_R`, `ROTOR_TIP_R` | `0.045`, `0.185` |
| half-length | `ROTOR_HALF_LEN` | `0.475` |
| blade thickness | `VANE_HALF_T` | `0.006` |
| boxes per blade | `VANE_SEGMENTS` | `5` |
| handedness | `RAKE_SIGN` | `1.0` |

Two constraints bound `ROTOR_VANES` and `ROTOR_RAKE` together, and
`check_excavator.py` fails on either:

- a blade sweeps `ln(TIP/HUB) * tan(rake)` radians and must fit inside one
  `360/vanes` pocket, or adjacent vanes shadow each other. Six vanes cap the
  rake near 35 degrees.
- the chord between adjacent vane tips is the way into a pocket, and it needs
  four grains. Six vanes give 0.185 m, which is 5.2 grains at a 0.03 voxel.
  Eight give 3.7 and arch.

## The shroud

| what | constant | now |
|---|---|---|
| inlet width | `SHROUD_INLET` | `radians(100)` |
| inner and outer radius | `SHROUD_IN_R`, `SHROUD_OUT_R` | `0.195`, `0.212` |
| end plate thickness | `SHROUD_END_T` | `0.012` |
| aiming range | `SHROUD_RANGE` | `(-1.10, 1.10)` |

`SHROUD_IN_R - ROTOR_TIP_R` is the running clearance, currently 10 mm nominal
and 7 mm measured. It has to be **smaller than one voxel** or the coupler leaves
it open and regolith escapes round the whole circumference instead of staying in
a pocket. Drop the voxel below 10 mm and this needs tightening with it.

## The machine

| what | constant | now |
|---|---|---|
| wheelbase, track | `WHEELBASE`, `TRACK` | `1.40`, `1.15` |
| wheel radius, width | `WHEEL_RADIUS`, `WHEEL_HALF_W` | `0.30`, `0.10` |
| grousers per wheel | `WHEEL_GROUSERS` | `6` (0 for a bare cylinder) |
| arm length | `ARM_LEN` | `0.76` |
| arm pivot ahead of the axle | `MAST_OFFSET_X` | `0.22` |
| arm travel | `ARM_RANGE` | `(-0.55, 0.72)` |

`ARM_RANGE[1]` is tied to `ROTOR_TIP_R`: 0.72 rad puts the vane tips 0.185 m
down, which is one rotor radius. Deeper and the vanes fight the whole overburden
instead of cutting.

## Rates and the solver

| what | where | now |
|---|---|---|
| wheel speed cap | `MAX_WHEEL_SPEED` in `excavator_cfg.py` | `5.0` rad/s |
| drum speed cap | `MAX_DRUM_SPEED` | `2.9` rad/s |
| contact buffers | `rigid_njmax`, `rigid_nconmax` | `8192`, `32768` |
| grid cells per particle | `grid_cap_multiplier` | `8.0` |
| cells per leaf node | `grid_leaf_shift` | `7` |

`MAX_DRUM_SPEED` is not a preference. Outward acceleration at the vane tips is
`omega^2 * ROTOR_TIP_R`, which passes lunar gravity at **2.96 rad/s**; above
that the vanes throw regolith at the shroud instead of carrying it inward. The
limit scales as `sqrt(g/r)`, so it moves if the rotor radius does.

Overrunning a contact buffer is an illegal memory access, not an error -- a CUDA
700 storm, or 0xC0000374 on Windows. A contact is a couple of hundred bytes, so
there is no reason to be tight.

## After changing anything in excavator.py

```powershell
.\isaaclab.bat -p -c "from luna_hifi_tasks.excavator.excavator import write_mjcf; print(write_mjcf())"
Remove-Item -Recurse -Force $REPO\assets\excavator -ErrorAction SilentlyContinue
.\isaaclab.bat -p scripts\tools\convert_mjcf.py $env:TEMP\luna_hifi_assets\excavator.xml $REPO\assets\excavator.usd
```

Check the geometry first; it takes two seconds and needs no GPU:

```powershell
& $REPO\.venv-geom\Scripts\python $REPO\scripts\check_excavator.py
```
