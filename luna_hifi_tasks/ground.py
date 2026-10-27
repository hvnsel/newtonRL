# ground.py
#
# The world ground plane, checked in rather than downloaded.
#
# isaaclab.sim.GroundPlaneCfg defaults usd_path to the Isaac asset server, so
# building a scene fetched
#
#   .../Isaac/6.2/Isaac/IsaacLab/Environments/Grid/
#       default_ground_plane_checker_v1/default_ground_plane.usda
#
# over HTTPS and only ever worked because an earlier download sat in the
# per-user cache under /tmp. Clearing that cache stranded three training
# launches in a row on
#
#   FileNotFoundError: Unable to find the file: https://...
#
# out of isaaclab/utils/assets.py, raised from spawn_ground_plane before a
# single step ran. That is a bad way for a run to die. A training box need not
# have outbound network, the asset server need not keep a 6.2-era path alive,
# and nothing about a flat collider should depend on either.
#
# So the asset lives in the repo, with the two textures it references beside
# it at the relative paths it names. It is the same USD the cache held, which
# matters: spawn_ground_plane reaches into this structure, binding the physics
# material to the first child Plane, scaling the Environment mesh to cfg.size
# and tinting Looks/theGrid/Shader. Keeping the file intact keeps the collider
# and the look identical to every run before this one -- only the fetch is
# gone.

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils

GROUND_PLANE_DIR = Path(__file__).resolve().parents[1] / "assets" / "ground_plane"
GROUND_PLANE_USD_PATH = GROUND_PLANE_DIR / "default_ground_plane.usda"


def ground_plane_spawn() -> sim_utils.GroundPlaneCfg:
    """GroundPlaneCfg pointed at the checked-in asset.

    Checked here rather than left to USD: an unresolved reference does not
    raise, it yields a prim with no children, and spawn_ground_plane then
    fails looking for the Plane to bind -- which reads as a USD bug rather
    than a missing file.
    """
    if not GROUND_PLANE_USD_PATH.is_file():
        raise FileNotFoundError(
            f"ground plane asset missing: {GROUND_PLANE_USD_PATH}\n"
            "  it is tracked in the repo; restore it with\n"
            "    git checkout -- assets/ground_plane"
        )
    return sim_utils.GroundPlaneCfg(usd_path=str(GROUND_PLANE_USD_PATH))
