# Running on a workstation

For a single Ubuntu box with its own GPU, no Slurm, no queue, root available.

Everything in `cluster/` exists to work around constraints that do not apply
here: a read-only `.sif`, scratch as the only writable path, a 12-hour wall, a
one-to-two-day queue, and no display. On your own machine the container is a
plain `docker run` with the checkout bind-mounted, and **an edit on the host is
live inside it**. No rebuild, no reinstall, no push-pull round trip.

The bigger change is the viewer. Every run so far has been read through
TensorBoard scalars, which is how a policy that creeps 2.6 m and stops took
five runs to characterise. Locally you can watch the drum cut soil.

---

## 1. Preflight

```bash
bash workstation/preflight.sh
```

Checks the driver, compute capability, Docker, the container toolkit, disk and
display, and prints the fix for anything missing. It changes nothing.

The two that actually block:

- **Driver.** Isaac Lab 3.0 recommends **580.65.06+**; the practical floor is
  around 535.129, below which Omniverse RTX refuses to start.
- **Compute capability ≥ 7.5.** The image's PyTorch has no kernels below it —
  this is what killed a cluster job that landed on a V100 (7.0). A 3090 is 8.6.

## 2. No root: uv, Python 3.12, and no Isaac Sim

If preflight says you are not in the `docker` group and have no sudo, this is
the path. It installs entirely under your home directory.

Isaac Lab 3.0's root `pyproject.toml` pins, verified against
`release/3.0.0`:

| | pin |
|---|---|
| Python | `>=3.12,<3.13` |
| Isaac Sim | `isaacsim[all,extscache]==6.2.0.0` — **optional**, see below |
| newton | `newton[sim]==1.6.1` |
| warp-lang | `1.17.0` |
| mujoco | `~=3.12.0`, plus `mujoco-warp~=3.12.0` |

Ubuntu 22.04 ships Python 3.10, which is outside that range, and `deadsnakes`
needs root. `uv` solves both problems: one binary into `~/.local/bin`, no root,
and it supplies its own interpreters. Isaac Lab's pyproject has a `[tool.uv]`
section with `override-dependencies`, so uv is the project's own tooling rather
than a workaround.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"          # add to ~/.bashrc
uv python install 3.12

git clone -b release/3.0.0 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
uv sync                                        # Kit-less: no Isaac Sim
```

**Try it without Isaac Sim first.** The 3.0.0 README states that Kit-less
Newton workflows do not require Isaac Sim, and this repo is a Newton workload:
MJWarp for the rigid tier, MPM for soil, and `drive_check.py` already defaults
to `visualizer=["newton_gl"]`, which is Newton's own GL viewer rather than Kit.
Skipping it avoids a 10+ GB download and the whole Kit shader-cache problem.

The parts that might still pull Kit in are the `DomeLightCfg` in
`NavigateSceneCfg` and loading `assets/excavator/excavator.usda`. If an import
fails on either, add Isaac Sim:

```bash
uv sync --extra isaacsim
```

`isaacsim` and `importers` are the two extra names in that pyproject. Confirm
what is on offer rather than guessing:

```bash
sed -n '/\[project.optional-dependencies\]/,/^\[/p' ~/IsaacLab/pyproject.toml
```

Then put this package in the same environment and run the port checks:

```bash
cd ~/newtonRL
uv pip install --python ~/IsaacLab/.venv/bin/python -e .
~/IsaacLab/.venv/bin/python scripts/run_task.py --task Luna-Excavator-Navigate --num_envs 4
```

Then use `workstation/train.sh` for everything after, which resolves the venv
interpreter and runs from the repo root so checkpoints land where peek.py
looks:

```bash
workstation/train.sh train --rl_library rsl_rl \
  --task Luna-Excavator-Navigate --num_envs 1024 --max_iterations 400
```

Do not use `isaaclab.sh`. It resolves its own interpreter, and where the system
python is older than the venv's it picks the wrong one and dies on `No module
named 'tomllib'` -- stdlib from 3.11 on, so that message means 3.10 ran it. It
is deprecated in favour of `uv run isaaclab`, which resolves the project from
the working directory and therefore wants to run from the Isaac Lab tree rather
than from here.

Two notes on version drift. The cluster ran `3.0.0-rc1`, whose compiled
bytecode in this checkout is `cpython-311`, so it was Python 3.11; final
`release/3.0.0` moved to 3.12 and Isaac Sim 6.2. The repo code is compatible
either way — `root_quat_w` is documented `(x, y, z, w)` in `release/3.0.0`,
which is what `yaw_from_quat` assumes. And the README warns `release/3.0.0` is
under active development and may break, so pin the clone to that tag rather
than tracking `develop`.

## 3. One-time setup, with Docker


```bash
# The image the cluster .sif was built from. ~25 GB unpacked.
docker pull nvcr.io/nvidia/isaac-lab:3.0.0-rc1

# Editable install of this package into a host-persisted --user prefix, so it
# survives the container exiting.
workstation/dev.sh isaaclab -p -m pip install --user -e .

# Offline geometry checks need only MuJoCo, so they run on the host.
pip install -e '.[assets]'
```

`dev.sh` keeps Kit's shader caches and that install in `$LUNA_STATE`
(default `~/.luna`). Without it every start pays the five-minute cold compile.

## 4. Verify the port

```bash
workstation/dev.sh isaaclab -p scripts/run_task.py --task Luna-Excavator-Navigate --num_envs 4
```

`run_task.py` asserts the gym ids register, the USD loads, and the observation
layout hashes match. The navigate actor spec should hash to `959c9a741a76`
with the scan and `e50b607a11db` without.

Geometry, with no container at all:

```bash
python scripts/check_excavator.py
```

## 5. Run

```bash
# navigate: measured 13.3 GB at 1024 envs, so this fits 24 GB with room
workstation/dev.sh isaaclab train --rl_library rsl_rl \
  --task Luna-Excavator-Navigate --num_envs 1024 --max_iterations 2000

# telemetry, from the host, no container needed
python scripts/peek.py --tags goal_dist_hi reached mean_std learning_rate

# navigate footage, headless: 12 one-episode clips over hazard terrain with the
# goal drawn, a trace per clip, and the graphs (scripts/plot_nav.py), in ~/nav_media
bash workstation/nav_videos.sh

# the obstacle course (mdp/course.py): goal straight ahead, craters or boulders
# on the line, one clip and graph per layout, in ~/nav_course
bash workstation/nav_course.sh

# watch it dig
xhost +local:docker
workstation/dev.sh isaaclab -p scripts/dig_demo.py
```

Expect roughly **1.5–1.8×** the cluster's per-iteration time on a 3090 against
an L40S: 142 SMs versus 82, though the 3090 has 8% *more* memory bandwidth
(936 vs 864 GB/s), so the 2.6× FP32 spec gap is not the real ratio. A
2000-iteration navigate run is ~13–15 h instead of 8.5 — against a queue of
zero.

## 6. Excavate: iterate on Micro, train on the full bed

Three beds are already registered, and the small ones are the reason excavate
iteration does not need a cluster:

| task | particles/env | MPM rounds on a 3090 | grid VRAM |
|---|---|---|---|
| `Luna-Excavator-Excavate` | 161,001 | 15.3 | 12–24 GB |
| `Luna-Excavator-Excavate-Small` | 6,882 | 0.66 | ~1 GB |
| `Luna-Excavator-Excavate-Micro` | 8,092 | **0.77** | **0.8–1.5 GB** |

Under one round means the whole particle set clears the GPU in a single pass,
so on Micro the MPM step stops being the bottleneck at all — the rigid solver
is. Against the full bed's 15.3 rounds that is roughly a 20× faster step, and
it needs no config changes: the grid fits even at the default `max_num_envs`.

Micro is a one-metre strip starting where the drum's leading edge reaches at
the dig angle, deep enough to bury 0.09 m of rotor — which is to say it is
exactly the geometry the dig reward is about, with none of the bed the machine
never touches. **This is where a "why won't it dig" question gets answered**,
with the viewer open, in seconds per run:

```bash
xhost +local:docker
workstation/dev.sh isaaclab -p scripts/dig_demo.py --task Luna-Excavator-Excavate-Micro
```

### The full bed on 24 GB

`excavate_env_cfg.py:480` sizes the MPM sparse grid from **`max_num_envs`, not
`--num_envs`**:

```python
total_particles = self.bed_particles_per_env * self.max_num_envs
active = _next_pow2(int(self.grid_cap_multiplier * total_particles))
```

At the default `max_num_envs = 104` that is `next_pow2(8 × 16.7M)` = **2²⁷ =
134M cells**, allocated whether you run 104 envs or 4 — 12–24 GB before
particles, the articulation, 32k contacts and the policy. So on 24 GB, lower it,
and keep it at or above the env count:

| `max_num_envs` | grid cells | grid VRAM |
|---|---|---|
| 104 (default) | 134M | 12–24 GB |
| 64 | 134M | identical — `next_pow2` quantises |
| **32** | 67M | **6–12 GB** |
| 16 | 34M | 3–6 GB |

```bash
workstation/dev.sh isaaclab train --rl_library rsl_rl \
  --task Luna-Excavator-Excavate --num_envs 32 env.max_num_envs=32
```

104 → 64 buys nothing. 32 is the first real step down.

The voxel lever is mostly closed, so do not reach for it: the 0.185 m vane-tip
gap is 5.2 grains clear at the 0.03 m voxel and 2.7 at 0.05, and granular
material arches below about four, making ~0.046 m the hard ceiling. 0.03 → 0.04
buys 2.3× fewer particles and spends most of that margin.

## 7. Excavate: what to look at first

Standing suspects for why the rover does not learn to dig, in the order the
viewer settles them fastest. All of these are Micro-bed questions.

1. **`soil_cohesion = 0.0`.** The config's own comment says that at zero
   cohesion the regolith behaves as dry sand and shears off the lip, while a
   few hundred Pa makes the cut travel as a clod the lifters carry round.
   Lunar simulants sit at 0.1–1 kPa. If the soil never forms a clod there may
   be nothing to learn. `dig_demo.py` takes it directly, so this is one command
   and no config edit:

   ```bash
   workstation/dev.sh isaaclab -p scripts/dig_demo.py \
     --task Luna-Excavator-Excavate-Micro --cohesion 300 --cut 0.10
   ```

   It also takes `--friction`, `--density`, `--cut`, `--drum` and `--drive`, so
   a sweep of the soil model is a shell loop rather than a rebuild.
2. **Spawn jitter against the bed edge.** `spawn_y_jitter = 0.55` m on a bed of
   `bed_y = (-1.0, 1.0)`, with the machine's own half-width around 0.6 m off a
   1.15 m track. The outer wheel can start past the bed edge. Ten seconds on
   the viewer settles whether it does.
3. **`reward_audit.py`** prints per-term episode totals over a zero pass and a
   random walk. Run it before changing any weight — the equivalent reading in
   navigate is what caught the goal bonus being paid twice.
4. **The multi-env sparse-grid capacity failure**, still unresolved, which is
   also why excavate's real multi-env footprint has never been measured.
