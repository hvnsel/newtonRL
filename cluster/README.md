# Running on PACE Phoenix

Account `gts-jmcnabb3`, 272 credits. Three filesystems, and picking the wrong
one is the first way this goes wrong:

| | size | persists? | what goes here |
|---|---|---|---|
| `$HOME` | **20 GB** | backed up | dotfiles, nothing else |
| `/storage/project/r-jmcnabb3-0` | 1 TB (32 GB used, **shared**) | yes | container image, repo, USD asset |
| `/storage/scratch1/5/$USER` | 15 TB | **purged** on access time | logs, checkpoints, caches |

An Isaac Sim install plus its caches does not fit in 20 GB. Omniverse and Warp
also ignore `XDG_CACHE_HOME` and write to fixed paths under `$HOME`.
`cluster/pace_env.sh` redirects all of it; source it in every shell and every
job. Skip it and the quota fills partway through a container pull, where it
looks like a corrupt download rather than a full disk.

## 1. Find out what the site actually offers

```bash
bash cluster/pace_probe.sh
```

Partition names, GPU types and how to request them, whether Apptainer is a
module — all site-specific. Fill them into `train.sbatch` from this output
rather than from anyone's assumptions.

The one fact it cannot get from a login node is the GPU driver version, which
is the thing most likely to veto an Isaac Sim image:

```bash
salloc -A gts-jmcnabb3 -N1 --gres=gpu:1 -t0:20:00
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

## 2. Lay it out

```bash
mkdir -p /storage/project/r-jmcnabb3-0/luna
cd /storage/project/r-jmcnabb3-0/luna
git clone <this repo> newtonRL
source newtonRL/cluster/pace_env.sh
```

## 3. The container

Isaac Lab is run from a container on clusters — a native install needs write
access to paths a shared filesystem will not give you, and pins a driver you do
not control. Pull it into project storage, not home:

```bash
source $LUNA_REPO/cluster/pace_env.sh
apptainer pull "$LUNA_SIF" docker://nvcr.io/nvidia/isaac-lab:<TAG>
```

Two things to settle before this will work, and both need checking rather than
guessing:

- **the tag.** It must be an Isaac Lab **3.x with the Newton backend** — this
  repo's MPM coupling does not exist on 2.x. Check what tags NGC publishes.
- **NGC auth.** Some NVIDIA images need an API key:
  `apptainer remote login --username '$oauthtoken' docker://nvcr.io`

If no published image carries Newton, the fallback is a definition file that
starts from the Isaac Sim base image and installs Isaac Lab's develop branch on
top. That is more work but it is the honest answer if the tag does not exist.

## 4. Build the excavator asset, once

`assets/excavator/` is generated and gitignored, so it does not arrive with the
clone. The converter needs Kit, so it runs inside the container:

```bash
apptainer exec --nv -B "$LUNA_PROJECT" -B "$LUNA_SCRATCH" "$LUNA_SIF" bash -lc '
  cd "$LUNA_REPO" && python -m pip install -e . &&
  python -c "from luna_hifi_tasks.excavator.excavator import write_mjcf; print(write_mjcf())" &&
  python /workspace/isaaclab/scripts/tools/convert_mjcf.py \
      "$TMPDIR/luna_hifi_assets/excavator.xml" "$LUNA_REPO/assets/excavator.usd"'
```

Output lands at `assets/excavator/excavator.usda` — note the path the converter
actually writes, which is not the one you pass it. `scripts/dig_demo.py` and
`scripts/smoke_test.py` check the asset against its source before spawning and
will tell you if it is missing or stale.

## 5. Check it without burning credits

`scripts/check_excavator.py` needs MuJoCo and **not** Isaac Lab, so it runs on a
login node in seconds:

```bash
pip install --user "mujoco>=3.13"    # ~50 MB, fits in home
python $LUNA_REPO/scripts/check_excavator.py
```

Then one short interactive job for the smoke test before anything long:

```bash
salloc -A gts-jmcnabb3 -N1 --gres=gpu:1 -t0:30:00
apptainer exec --nv ... python scripts/smoke_test.py --task Luna-Excavator-Navigate --num_envs 8
```

## 6. Sizing, once a GPU is in hand

The laptop numbers do not transfer and should not be copied across:

- `nconmax` / `njmax` are per-env fixed buffers in the rigid solver. Overrunning
  one is an illegal access, not an error. 8192/4096 here.
- the MPM sparse-grid caps are absolute totals across all envs and do **not**
  scale with `--num_envs`. `max_num_envs` on the cfg is what they are sized for,
  and the env asserts against it at start-up.
- `voxel_size` decides whether soil can enter the drum at all: the coupler eats
  a whole voxel out of every passage and particles sit one voxel apart. The
  drum's entry channel opens 0.096 m, so 0.02 gives 3.8 grain diameters and
  0.03 gives 2.2. Granular material arches below about four.

Raise `max_num_envs` and the caps together, and re-check with one short job
before committing to a long one.
