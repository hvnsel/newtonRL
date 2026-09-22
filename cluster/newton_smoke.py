# cluster/newton_smoke.py -- does this container have a working Newton stack?
#
# Standalone on purpose: no luna_hifi_tasks, no repo, no credentials. Copy this
# ONE file to the cluster and run it inside the container. It answers the only
# question that matters before any training code exists --
#
#     is the dependency chain that this project needs actually present and
#     functional on this GPU?
#
#   apptainer exec --nv image.sif python newton_smoke.py
#
# It is LAYERED rather than pass/fail. Each layer prints what it found and
# carries on where it can, because "isaaclab imports but isaaclab_newton does
# not" and "everything imports but no GPU is visible" are completely different
# problems with completely different fixes, and a single traceback tells you
# which one only by accident.

from __future__ import annotations

import os
import platform
import sys

OK, BAD, WARN = "  ok   ", " FAIL  ", " warn  "
findings: list[str] = []


def check(label: str, fn):
    """Run one probe. Returns its value, or None if it raised."""
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - reporting is the whole point
        print(f"{BAD}{label}: {type(exc).__name__}: {exc}")
        findings.append(f"{label}: {type(exc).__name__}: {exc}")
        return None
    print(f"{OK}{label}: {value}")
    return value


print("=== host ===")
print(f"  python   {platform.python_version()} at {sys.executable}")
print(f"  platform {platform.platform()}")
print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")

print("\n=== layer 1: torch and the GPU ===")
torch = check("import torch", lambda: __import__("torch").__version__)
if torch is not None:
    import torch as t

    avail = check("cuda available", lambda: t.cuda.is_available())
    if avail:
        check("device", lambda: t.cuda.get_device_name(0))
        check("capability", lambda: ".".join(map(str, t.cuda.get_device_capability(0))))
        check("VRAM (GB)", lambda: round(t.cuda.get_device_properties(0).total_memory / 1e9, 1))
        check("torch CUDA build", lambda: t.version.cuda)
    else:
        findings.append("no CUDA device visible -- are you on a GPU node, and did "
                        "apptainer get --nv?")

print("\n=== layer 2: warp (Newton's kernel layer) ===")
wp = check("import warp", lambda: __import__("warp").__version__)
if wp is not None:
    import warp

    def _warp_device():
        warp.init()
        devs = [str(d) for d in warp.get_devices()]
        return f"{devs} (default {warp.get_preferred_device()})"

    check("warp devices", _warp_device)
    check("warp kernel cache", lambda: warp.config.kernel_cache_dir)

print("\n=== layer 3: isaaclab ===")
check("import isaaclab", lambda: __import__("isaaclab").__version__)
check("isaaclab.sim", lambda: __import__("isaaclab.sim", fromlist=["x"]).__name__)

print("\n=== layer 4: the Newton backend (this is the one that matters) ===")
# 2.x does not have these at all. If this layer fails, the image is the wrong
# Isaac Lab line and nothing downstream will work, however green layers 1-3 are.
check("isaaclab_newton", lambda: __import__("isaaclab_newton").__name__)
check("MPMObjectCfg", lambda: __import__(
    "isaaclab_newton.assets", fromlist=["MPMObjectCfg"]).MPMObjectCfg.__name__)
check("MPMSolverCfg", lambda: __import__(
    "isaaclab_newton.physics", fromlist=["MPMSolverCfg"]).MPMSolverCfg.__name__)
check("NewtonCfg", lambda: __import__(
    "isaaclab_newton.physics", fromlist=["NewtonCfg"]).NewtonCfg.__name__)
check("MJWarpSolverCfg", lambda: __import__(
    "isaaclab_newton.physics", fromlist=["MJWarpSolverCfg"]).MJWarpSolverCfg.__name__)
check("MPMParticleMaterialCfg", lambda: __import__(
    "isaaclab_newton.sim.spawners.mpm", fromlist=["MPMParticleMaterialCfg"]
).MPMParticleMaterialCfg.__name__)

print("\n=== layer 5: rigid <-> MPM coupling ===")
# The excavator needs this specifically: two solvers sharing contacts.
check("CouplerProxyCfg", lambda: __import__(
    "isaaclab_contrib.coupling", fromlist=["CouplerProxyCfg"]).CouplerProxyCfg.__name__)
check("CouplerEntryCfg", lambda: __import__(
    "isaaclab_contrib.coupling", fromlist=["CouplerEntryCfg"]).CouplerEntryCfg.__name__)

print("\n=== layer 6: does a kit app actually start? ===")
# Everything above is imports. This is the first thing that touches the GPU
# driver and the EULA, and it is where a headless node usually objects.
def _launch():
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    ver = getattr(app, "version", "started")
    app.close()
    return ver

check("AppLauncher(headless=True)", _launch)

print("\n=== verdict ===")
if not findings:
    print("  everything this project needs is present.")
    print("  Next: copy the training code in and run its own smoke test.")
else:
    for f in findings:
        print(f"  UNRESOLVED: {f}")
    print("\n  Read them in order -- layer 4 failing means the image is the wrong")
    print("  Isaac Lab line (Newton arrived in 3.x), which no amount of fixing")
    print("  layers 5 and 6 will help with.")
sys.exit(1 if findings else 0)
