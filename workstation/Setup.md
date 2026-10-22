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

## 2. One-time setup

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

## 3. Verify the port

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

## 4. Run

```bash
# navigate: measured 13.3 GB at 1024 envs, so this fits 24 GB with room
workstation/dev.sh isaaclab train --rl_library rsl_rl \
  --task Luna-Excavator-Navigate --num_envs 1024 --max_iterations 2000

# telemetry, from the host, no container needed
python scripts/peek.py --tags goal_dist_hi reached mean_std learning_rate

# watch it dig
xhost +local:docker
workstation/dev.sh isaaclab -p scripts/dig_demo.py
```

Expect roughly **1.5–1.8×** the cluster's per-iteration time on a 3090 against
an L40S: 142 SMs versus 82, though the 3090 has 8% *more* memory bandwidth
(936 vs 864 GB/s), so the 2.6× FP32 spec gap is not the real ratio. A
2000-iteration navigate run is ~13–15 h instead of 8.5 — against a queue of
zero.

## 5. Excavate: iterate on Micro, train on the full bed

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

## 6. Excavate: what to look at first

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
