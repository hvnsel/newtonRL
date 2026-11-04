# The world ground plane, from the copy of Isaac Lab's default ground plane
# USD (and its textures) in assets/ground_plane/.

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils

GROUND_PLANE_USD_PATH = Path(__file__).resolve().parents[1] / "assets" / "ground_plane" / "default_ground_plane.usda"


def ground_plane_spawn() -> sim_utils.GroundPlaneCfg:
    """GroundPlaneCfg pointed at the checked-in asset."""
    return sim_utils.GroundPlaneCfg(usd_path=str(GROUND_PLANE_USD_PATH))
