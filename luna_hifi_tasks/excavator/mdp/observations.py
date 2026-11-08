# Observation layouts and scan geometry.

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

# Navigate scans: square grids centred on the chassis, yaw-aligned.
# far: 27 x 27 at 0.45 m, 5.85 m out in every direction.
# near: 25 x 25 at 0.20 m, 2.4 m out.
NAV_FAR_NX, NAV_FAR_NY = 27, 27
NAV_FAR_CELL = 0.45
NAV_FAR_CELLS = NAV_FAR_NX * NAV_FAR_NY

NAV_NEAR_NX, NAV_NEAR_NY = 25, 25
NAV_NEAR_CELL = 0.20
NAV_NEAR_CELLS = NAV_NEAR_NX * NAV_NEAR_NY

NAV_SCAN_CELLS = NAV_FAR_CELLS + NAV_NEAR_CELLS

# The near scan as a (C, H, W) = (1, NY, NX) grid: rows run along body y,
# columns along body x.
NAV_NEAR_IMAGE = (1, NAV_NEAR_NY, NAV_NEAR_NX)

# GridPatternCfg sizes giving exactly NX x NY rays, a ray at both ends of each axis.
NAV_FAR_SIZE = ((NAV_FAR_NX - 1) * NAV_FAR_CELL, (NAV_FAR_NY - 1) * NAV_FAR_CELL)
NAV_NEAR_SIZE = ((NAV_NEAR_NX - 1) * NAV_NEAR_CELL, (NAV_NEAR_NY - 1) * NAV_NEAR_CELL)

# Excavate critic's clean scan: forward-biased windows ahead of the chassis.
DIG_CRITIC_FAR_NX, DIG_CRITIC_FAR_NY = 16, 8
DIG_CRITIC_FAR_CELL = 0.45
DIG_CRITIC_FAR_BIAS = 2.40
DIG_CRITIC_NEAR_NX, DIG_CRITIC_NEAR_NY = 12, 8
DIG_CRITIC_NEAR_CELL = 0.20
DIG_CRITIC_NEAR_BIAS = 1.30
DIG_CRITIC_SCAN_CELLS = DIG_CRITIC_FAR_NX * DIG_CRITIC_FAR_NY + DIG_CRITIC_NEAR_NX * DIG_CRITIC_NEAR_NY

# Excavate actor's scan: the cutting face at the front drum.
DIG_SCAN_NX, DIG_SCAN_NY = 16, 8
DIG_SCAN_CELL = 0.125
DIG_SCAN_CELLS = DIG_SCAN_NX * DIG_SCAN_NY


@dataclass(frozen=True)
class ObsTerm:
    """One named block of the observation vector."""

    name: str
    dim: int
    note: str = ""


class ObsSpec:
    """An ordered list of observation terms."""

    def __init__(self, terms: list[ObsTerm]) -> None:
        self.terms = list(terms)

    @property
    def dim(self) -> int:
        return sum(t.dim for t in self.terms)

    def schema_hash(self) -> str:
        """12-char digest of the (name, dim) pairs in order."""
        payload = "|".join(f"{t.name}:{t.dim}" for t in self.terms)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def slice_of(self, name: str) -> slice:
        """Where a term sits in the assembled vector."""
        off = 0
        for t in self.terms:
            if t.name == name:
                return slice(off, off + t.dim)
            off += t.dim
        raise KeyError(name)

    def assemble(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate the (E, dim) parts in declared order."""
        return torch.cat([parts[t.name] for t in self.terms], dim=-1)

    def describe(self) -> str:
        lines = [f"ObsSpec  dim={self.dim}  hash={self.schema_hash()}"]
        off = 0
        for t in self.terms:
            lines.append(f"  [{off:>4}:{off + t.dim:>4}]  {t.name:<24} {t.dim:>4}   {t.note}")
            off += t.dim
        return "\n".join(lines)


def navigate_obs_spec(num_path_points: int = 2, num_wheels: int = 4) -> ObsSpec:
    """Navigate's "policy" group, the actor's whole observation."""
    return ObsSpec([
        ObsTerm("base_lin_vel", 3, "body frame"),
        ObsTerm("base_ang_vel", 3, "body frame"),
        ObsTerm("projected_gravity", 3, "body frame"),
        ObsTerm("path_points", 2 * num_path_points, "planned path points ahead, body-frame xy [m]"),
        ObsTerm("goal_vec_b", 2, "goal xy in body frame / goal_dist_max"),
        ObsTerm("goal_vec_near", 2, "goal xy in body frame / max(dist, goal_near_range)"),
        ObsTerm("goal_heading", 2, "(sin, cos) of twice the heading error"),
        ObsTerm("wheel_vel", num_wheels, "joint velocities, scaled"),
        ObsTerm("last_action", 2, "[forward, yaw]"),
    ])


def excavate_obs_spec(terrain_cells: int = DIG_SCAN_CELLS, num_arms: int = 2, num_wheels: int = 4) -> ObsSpec:
    """Excavate's observation.

    terrain_scan: 2.0 x 1.0 m at 0.125 m, centred on the front drum.
    drum_phase: (sin, cos) of vane count times rotor angle.
    target_level: drum height minus target-plane height, in terrain_scan's
    frame; grad_forward and grad_lateral: the plane's rise over run along
    body x and y.
    """
    return ObsSpec([
        ObsTerm("base_lin_vel", 3, "body frame"),
        ObsTerm("base_ang_vel", 3, "body frame"),
        ObsTerm("projected_gravity", 3, "body frame"),
        ObsTerm("wheel_vel", num_wheels, ""),
        ObsTerm("arm_pos", num_arms, "boom angle"),
        ObsTerm("arm_vel", num_arms, ""),
        ObsTerm("drum_vel", num_arms, ""),
        ObsTerm("shroud_pos", num_arms, "inlet angle"),
        ObsTerm("drum_phase", 2 * num_arms, "(sin, cos) of vanes * rotor angle"),
        ObsTerm("arm_torque", num_arms, "joint torque / ARM_EFFORT"),
        ObsTerm("drum_torque", num_arms, "joint torque / DRUM_EFFORT"),
        ObsTerm("drum_fill", num_arms, "estimated load / target_load_kg"),
        ObsTerm("target_level", 1, "drum height minus plane height"),
        ObsTerm("grad_forward", 1, "rise/run along body +x"),
        ObsTerm("grad_lateral", 1, "rise/run along body +y"),
        ObsTerm("depth_error", 1, "ground minus target over the footprint"),
        ObsTerm("cut_progress", 1, "fraction of the footprint at or below target"),
        ObsTerm("terrain_scan", terrain_cells, "drum-centred heights"),
        ObsTerm("last_action", 2, "[forward, yaw]"),
    ])


def critic_state_spec(terrain_cells: int = NAV_SCAN_CELLS, num_arms: int = 2) -> ObsSpec:
    """Privileged critic state ("critic" group / state_space)."""
    return ObsSpec([
        ObsTerm("terrain_scan_true", terrain_cells, "noise-free scan"),
        ObsTerm("drum_fill_mass", num_arms, "kg, exact"),
        ObsTerm("base_lin_vel_w", 3, "world frame"),
        ObsTerm("slip", num_arms, "commanded vs achieved contact speed"),
    ])
