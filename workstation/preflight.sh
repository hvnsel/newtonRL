#!/bin/bash
# Check an Ubuntu box can run the Isaac Lab container before anything else is
# installed. Reports every problem it finds rather than stopping at the first.
#
#   bash workstation/preflight.sh
#
# Nothing here changes the machine.

set -u
fail=0
# $USER is not set in every shell, and set -u makes that fatal mid-report.
WHO="${USER:-$(id -un)}"
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mBAD\033[0m   %s\n' "$1"; fail=$((fail+1)); }
note() { printf '        %s\n' "$1"; }

echo
echo "=== GPU and driver ==="
if ! command -v nvidia-smi >/dev/null; then
  bad "nvidia-smi not found: no NVIDIA driver installed"
else
  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  ok "$GPU, ${VRAM} MiB, driver $DRV"

  # Isaac Lab 3.0 recommends 580.65.06+; reports put the practical floor near
  # 535.129, below which Omniverse RTX refuses to start.
  MAJ=${DRV%%.*}
  if [ "${MAJ:-0}" -ge 580 ] 2>/dev/null; then
    ok "driver $DRV meets the 580.65.06 recommendation"
  elif [ "${MAJ:-0}" -ge 535 ] 2>/dev/null; then
    note "driver $DRV is below the recommended 580.65.06. It may work -- the"
    note "practical floor is around 535.129 -- but upgrade if Kit fails to start:"
    note "  sudo apt install nvidia-driver-580   # then reboot"
  else
    bad "driver $DRV is too old for Omniverse RTX; install 580 or newer"
  fi

  # The image's PyTorch needs compute capability 7.5+. This is what sent a
  # cluster job to its death on a V100 (7.0). A 3090 is 8.6.
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "${CC:-}" ]; then
    if awk "BEGIN{exit !($CC >= 7.5)}"; then
      ok "compute capability $CC (needs >= 7.5)"
    else
      bad "compute capability $CC is below 7.5: the image has no kernels for this GPU"
    fi
  fi

  if [ "${VRAM:-0}" -lt 20000 ] 2>/dev/null; then
    note "under 20 GB of VRAM. Navigate measured 13.3 GB at 1024 envs, so it"
    note "fits; excavate needs max_num_envs lowered. See workstation/Setup.md."
  fi
fi

echo
echo "=== root ==="
# On a managed lab machine this is the question that decides everything below:
# adding yourself to the docker group needs root, and without it Docker is not
# available to you at all no matter that it is installed and running.
HAVE_SUDO=no
if [ "$(id -u)" -eq 0 ]; then
  HAVE_SUDO=yes; ok "running as root"
elif sudo -n true 2>/dev/null; then
  HAVE_SUDO=yes; ok "passwordless sudo"
elif sudo -nv 2>&1 | grep -q 'password'; then
  HAVE_SUDO=maybe; ok "sudo available (will prompt for a password)"
else
  note "no sudo. Docker needs group membership an admin has to grant, so the"
  note "rootless paths below are the ones open to you."
fi

echo
echo "=== Docker and the container toolkit ==="
if ! command -v docker >/dev/null; then
  bad "docker not found:  sudo apt install docker.io"
else
  ok "docker $(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  if ! docker info >/dev/null 2>&1; then
    if id -nG | tr ' ' '\n' | grep -qx docker; then
      bad "in the docker group but still refused: the group is not active in"
      note "this shell. Log out and back in, or:  exec newgrp docker"
    elif [ "$HAVE_SUDO" = no ]; then
      bad "not in the docker group, and no sudo to add yourself. Ask whoever"
      note "administers this box to run:"
      note "  sudo usermod -aG docker ${WHO}"
      note "Until then, see the rootless fallbacks printed at the end."
    else
      bad "not in the docker group. Add yourself, then log out and back in:"
      note "  sudo usermod -aG docker ${WHO}"
      note "  exec newgrp docker        # or just reconnect"
      note "If the daemon is also stopped:  sudo systemctl enable --now docker"
    fi
  else
    ok "docker daemon reachable without sudo"
    # The real test: does a container see the GPU? Isaac Sim needs
    # nvidia-container-toolkit 1.17.0+.
    if docker run --rm --gpus all "${IMAGE_PROBE:-ubuntu:22.04}" nvidia-smi -L >/dev/null 2>&1; then
      ok "containers can see the GPU (nvidia-container-toolkit works)"
    else
      bad "a container cannot see the GPU. Install nvidia-container-toolkit 1.17.0+:"
      note "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \\"
      note "    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
      note "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\"
      note "    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\"
      note "    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
      note "  sudo apt update && sudo apt install -y nvidia-container-toolkit"
      note "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi
  fi
fi

echo
echo "=== disk ==="
# The image unpacks to ~25 GB; logs and checkpoints grow from there.
for d in /var/lib/docker "$HOME"; do
  AVAIL=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "${AVAIL:-}" ] || continue
  if [ "$AVAIL" -ge 60 ]; then ok "$d has ${AVAIL}G free"
  elif [ "$AVAIL" -ge 30 ]; then note "$d has only ${AVAIL}G free; the image alone wants ~25G"
  else bad "$d has only ${AVAIL}G free; need 60G for the image plus logs"; fi
done

echo
echo "=== display, for the viewer ==="
if [ -n "${DISPLAY:-}" ] && [ -S "/tmp/.X11-unix/X${DISPLAY#*:}" ] 2>/dev/null; then
  ok "DISPLAY=$DISPLAY with a local X socket: dev.sh will forward it"
elif [ -n "${DISPLAY:-}" ]; then
  note "DISPLAY=$DISPLAY but no local X socket. Over SSH the viewer will not"
  note "open; headless training is unaffected."
else
  note "no DISPLAY. Headless training works; the viewer needs X on the box"
  note "itself, or x11-forwarding, or run the GUI from a local session."
fi

echo
echo "=== rootless fallbacks ==="
# Only worth reading if Docker came back refused. Both of these run entirely
# as you, with no group membership and no daemon.
if command -v apptainer >/dev/null || command -v singularity >/dev/null; then
  APP=$(command -v apptainer || command -v singularity)
  ok "$APP present: runs as you, no daemon, no group"
  note "This is the cluster's own path. Build the image once, then run it:"
  note "  apptainer build ~/isaaclab.sif docker://nvcr.io/nvidia/isaac-lab:3.0.0-rc1"
  note "Expect ~45 min and 70 GB of scratch while it builds. cluster/build_sif.sbatch"
  note "is the same recipe, and IMAGE=~/isaaclab.sif is understood by dev.sh only"
  note "after swapping docker run for apptainer exec -- see Setup.md."
else
  note "no apptainer. It needs root to install, but once installed it runs"
  note "rootless:  sudo apt install apptainer"
fi
note ""
note "Or skip containers entirely -- Isaac Sim pip-installs into a venv in your"
note "home directory and needs no root at all:"
note "  python3 -m venv ~/isaacenv && . ~/isaacenv/bin/activate"
note "  pip install --upgrade pip"
note "  pip install 'isaacsim[all,extscache]==5.0.0' --extra-index-url https://pypi.nvidia.com"
note "Then clone Isaac Lab 3.0 and run its ./isaaclab.sh --install. This is the"
note "best path for iteration anyway: no container boundary, native viewer."

echo
if [ "$fail" -eq 0 ]; then
  echo "preflight passed. Next: workstation/Setup.md"
else
  echo "$fail blocking problem(s) above."
fi
exit "$fail"
