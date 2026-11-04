# Running on a workstation

One Ubuntu box with its own GPU, headless. Either an Isaac Lab venv built with
uv (no root needed) or the Isaac Lab container.

Requirements:

- NVIDIA driver 580.65.06+ recommended; Omniverse RTX does not start below
  about 535.129.
- GPU compute capability 7.5 or higher (a 3090 is 8.6).

## 1. Isaac Lab venv with uv

Isaac Lab `release/3.0.0` pins:

| | pin |
|---|---|
| Python | `>=3.12,<3.13` |
| Isaac Sim | `isaacsim[all,extscache]==6.2.0.0` (optional extra) |
| newton | `newton[sim]==1.6.1` |
| warp-lang | `1.17.0` |
| mujoco | `~=3.12.0`, `mujoco-warp~=3.12.0` |

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"          # also in ~/.bashrc
uv python install 3.12

git clone -b release/3.0.0 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
uv sync                                        # kit-less, no Isaac Sim
uv sync --extra isaacsim                       # Isaac Sim, for the MJCF -> USD converter

cd ~/newtonRL
uv pip install --python ~/IsaacLab/.venv/bin/python -e .
```

`workstation/train.sh <args>` runs the Isaac Lab CLI with that venv from the
repo root.

## 2. Or the Isaac Lab container

```bash
docker pull nvcr.io/nvidia/isaac-lab:3.0.0-rc1
workstation/dev.sh isaaclab -p -m pip install --user -e .
```

`workstation/dev.sh isaaclab <args>` takes the same arguments as `train.sh`.
Kit caches and the install persist in `$LUNA_STATE` (default `~/.luna`).

## 3. Excavator asset

Build it once (README.md, "Assets").

## 4. Run

```bash
# navigate: 256 envs, 4000 iterations, a checkpoint every 25
workstation/train.sh train --rl_library rsl_rl \
  --task Luna-Excavator-Navigate --num_envs 256 > train.log 2>&1 &

# scalars from the newest run
python scripts/peek.py --tags goal_dist_hi reached mean_std learning_rate

# 12 one-episode clips on hazard terrain, traces and figures, in ~/nav_media
bash workstation/nav_videos.sh

# the obstacle course (mdp/course.py), one clip and figure per layout, in ~/nav_course
bash workstation/nav_course.sh
```

## 5. Excavate

| task | bed | envs |
|---|---|---|
| `Luna-Excavator-Excavate` | 4.56 × 1.6 × 0.25 m, machine on it | `max_num_envs` = 4 |
| `Luna-Excavator-Excavate-Small` | 0.9 × 1.1 × 0.16 m pile ahead of the machine | 1 |
| `Luna-Excavator-Excavate-Micro` | 1.0 × 1.0 × 0.21 m pile ahead of the machine | 1 |

The MPM sparse grid is sized from `max_num_envs` in `__post_init__`, not from
`--num_envs`; `--num_envs` above it stops the env. More envs need a larger
`max_num_envs` in `excavate_env_cfg.py`.

```bash
workstation/train.sh train --rl_library rsl_rl --task Luna-Excavator-Excavate --num_envs 4

# scripted dig on the Micro pile, soil overrides as flags
workstation/train.sh -p scripts/dig_demo.py --task Luna-Excavator-Excavate-Micro \
  --no_window --cohesion 300 --cut 0.10
```
