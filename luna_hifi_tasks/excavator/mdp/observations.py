# observations.py

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Scan geometry. The observation width is derived from it and the terrain
# sensor builds its patterns from it.
# ---------------------------------------------------------------------------

# Navigation, there are two resolutions: a coarse far one for routing and a
# fine near one for wheel placement. Both are square and centred on the
# chassis, so the machine sees as far behind and to either side as ahead: it
# has no front. The rear assembly is the front turned 180 degrees, and the dig
# leads with whichever drum faces the way it drives. Being centred is what
# makes turning the machine round map each grid onto itself (see
# mdp/symmetry.py); the odd counts put a cell under the chassis centre.
#
# far: 27 x 27 at 0.45 m, 5.85 m out in every direction, 4.1 m past either
# drum. near: 25 x 25 at 0.20 m, 2.4 m out, one cell per 0.20 m wheel width.
NAV_FAR_NX, NAV_FAR_NY = 27, 27
NAV_FAR_CELL = 0.45
NAV_FAR_BIAS = 0.0
NAV_FAR_CELLS = NAV_FAR_NX * NAV_FAR_NY

NAV_NEAR_NX, NAV_NEAR_NY = 25, 25
NAV_NEAR_CELL = 0.20
NAV_NEAR_BIAS = 0.0
NAV_NEAR_CELLS = NAV_NEAR_NX * NAV_NEAR_NY

# Total navigation scan width, which is also what the critic gets clean.
NAV_SCAN_CELLS = NAV_FAR_CELLS + NAV_NEAR_CELLS

# The navigate actor reads the two windows as single-channel images, (C, H, W)
# = (1, NY, NX). The ray caster's "xy" grid lays its rays out with x fastest,
# so a reshape puts y down the rows, from the machine's right to its left, and
# x along the columns, from behind it to ahead of it: a picture of the ground.
NAV_FAR_IMAGE = (1, NAV_FAR_NY, NAV_FAR_NX)
NAV_NEAR_IMAGE = (1, NAV_NEAR_NY, NAV_NEAR_NX)

# The dig critic's clean scan around the chassis. It was taken with the
# navigate windows' old forward-biased geometry and keeps it, so the excavate
# critic does not change shape with navigate's sensors.
DIG_CRITIC_FAR_NX, DIG_CRITIC_FAR_NY = 16, 8
DIG_CRITIC_FAR_CELL = 0.45
DIG_CRITIC_FAR_BIAS = 2.40
DIG_CRITIC_NEAR_NX, DIG_CRITIC_NEAR_NY = 12, 8
DIG_CRITIC_NEAR_CELL = 0.20
DIG_CRITIC_NEAR_BIAS = 1.30
DIG_CRITIC_SCAN_CELLS = DIG_CRITIC_FAR_NX * DIG_CRITIC_FAR_NY + DIG_CRITIC_NEAR_NX * DIG_CRITIC_NEAR_NY

# Excavation: the cutting face, at the front drum, which is the one meeting
# fresh soil under symmetric arm control.
DIG_SCAN_NX, DIG_SCAN_NY = 16, 8
DIG_SCAN_CELL = 0.125
DIG_SCAN_CELLS = DIG_SCAN_NX * DIG_SCAN_NY

# GridPatternCfg puts a point at both ends of each axis:
# arange(-size/2, size/2 + eps, res) yields size/res + 1 points, so a pattern
# size of (NX-1)*cell gives exactly NX x NY rays.
NAV_FAR_SIZE = ((NAV_FAR_NX - 1) * NAV_FAR_CELL, (NAV_FAR_NY - 1) * NAV_FAR_CELL)
NAV_NEAR_SIZE = ((NAV_NEAR_NX - 1) * NAV_NEAR_CELL, (NAV_NEAR_NY - 1) * NAV_NEAR_CELL)


@dataclass(frozen=True)
class ObsTerm:
    """One named block of the observation vector."""

    name: str
    dim: int
    note: str = ""

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"observation term {self.name!r} has non-positive dim {self.dim}")


class ObsSpec:
    """An ordered list of observation terms. Order is part of the layout the
    digest covers."""

    def __init__(self, terms: list[ObsTerm]) -> None:
        names = [t.name for t in terms]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate observation terms: {sorted(dupes)}")
        self.terms = list(terms)

    @property
    def dim(self) -> int:
        return sum(t.dim for t in self.terms)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.terms]

    def schema_hash(self) -> str:
        """Stable 12-char digest of (name, dim) pairs in order. `note` is
        excluded."""
        payload = "|".join(f"{t.name}:{t.dim}" for t in self.terms)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def slice_of(self, name: str) -> slice:
        """Where a term sits in the assembled vector."""
        off = 0
        for t in self.terms:
            if t.name == name:
                return slice(off, off + t.dim)
            off += t.dim
        raise KeyError(f"no observation term named {name!r}; have {self.names}")

    def assemble(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate terms in declared order, checking every shape.

        Raises on a missing term, an extra term, or a width mismatch.
        """
        missing = [t.name for t in self.terms if t.name not in parts]
        if missing:
            raise KeyError(f"observation terms not provided: {missing}")
        extra = sorted(set(parts) - set(self.names))
        if extra:
            raise KeyError(f"observation parts not in the spec: {extra}")

        ordered = []
        for t in self.terms:
            x = parts[t.name]
            if x.ndim != 2:
                raise ValueError(
                    f"observation term {t.name!r} must be (num_envs, dim), got {tuple(x.shape)}"
                )
            if x.shape[1] != t.dim:
                raise ValueError(
                    f"observation term {t.name!r} declares dim {t.dim} but got {x.shape[1]}"
                )
            ordered.append(x)

        out = torch.cat(ordered, dim=-1)
        if out.shape[1] != self.dim:
            raise AssertionError(f"assembled {out.shape[1]} columns, spec says {self.dim}")
        return out

    def describe(self) -> str:
        lines = [f"ObsSpec  dim={self.dim}  hash={self.schema_hash()}"]
        off = 0
        for t in self.terms:
            lines.append(f"  [{off:>4}:{off + t.dim:>4}]  {t.name:<24} {t.dim:>4}"
                         + (f"   {t.note}" if t.note else ""))
            off += t.dim
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ObsSpec(dim={self.dim}, hash={self.schema_hash()}, terms={self.names})"

def navigate_obs_spec(num_arms: int = 2, num_wheels: int = 4) -> ObsSpec:
    """The navigation skill's 1-D observation, the "policy" group.

    The terrain is not in it. The actor gets two windows as image groups of
    their own, "far_scan" (NAV_FAR_IMAGE) and "near_scan" (NAV_NEAR_IMAGE),
    each through a small CNN encoder trained with the policy; see
    navigate/agents/rsl_rl_ppo_cfg.py. Both are centred on the chassis:
    far_scan reaches 4.1 m past either drum, at the scale of a route through
    the terrain; near_scan resolves one wheel width, at the scale of a single
    wheel placement. A cell the sensor did not measure reads scan_invalid,
    which sits outside the clipped range of a real height.

    goal_heading is the goal's axis rather than its direction: either end of
    the machine may face along it, so the angle is doubled before (sin, cos),
    which reads the same for the machine turned 180 degrees.
    """
    terms = [
        ObsTerm("base_lin_vel", 3, "body frame"),
        ObsTerm("base_ang_vel", 3, "body frame"),
        ObsTerm("projected_gravity", 3, "body frame; also the tip-over signal"),
        ObsTerm("goal_vec_b", 2, "goal xy in BODY frame, scaled"),
        ObsTerm("goal_vec_near", 2, "goal xy in BODY frame / max(dist, goal_near_range)"),
        ObsTerm("goal_heading", 2, "(sin, cos) of twice the heading error: the axis"),
        ObsTerm("wheel_vel", num_wheels, "joint velocities, scaled"),
        ObsTerm("arm_pos", num_arms, "held stowed here, but it moves the CG"),
        ObsTerm("drum_fill", num_arms, "0 on the rigid tier; keeps the layout"),
        ObsTerm("last_action", 2, "[forward, yaw]"),
    ]
    # Concatenated in here, the forward scans were 224 of 249 inputs against
    # the 6 that carry the goal, and the actor had to find the goal underneath
    # them. Through the encoders the centred windows, 1354 cells, reach the
    # MLP as 288 numbers.
    return ObsSpec(terms)


def excavate_obs_spec(
    terrain_cells: int = DIG_SCAN_CELLS,
    num_arms: int = 2,
    num_wheels: int = 4,
) -> ObsSpec:
    """Observation for the excavation skill.

    `terrain_scan` is a 2-D window, 2.0 x 1.0 m at 0.125 m, centred on the
    front drum. It carries the approach angle relative to the face and the
    lateral slope.

    `drum_phase` is (sin, cos) of the vane count times the rotor angle, so it
    reads phase within a pocket and is identical for every pocket. Six vanes
    at 2.9 rad/s put a pocket mouth at the inlet every 0.36 s, 9 steps at
    25 Hz.

    `arm_torque` is the load sensor. It weighs what the arm carries, which is
    the soil held clear of the ground.

    The cut command is a plane, in three numbers. The rotor is a cylinder, so
    its cutting edge is a straight line across the swath at a constant depth,
    and the machine ramps along its direction of travel by raising the boom as
    it advances. `target_level` is in the same frame and units as every cell
    of terrain_scan, height relative to the front drum clipped to scan_clip;
    the two gradients are dimensionless rise over run in the body frame.

    `grad_lateral` is the component the drum meets across its swath, and the
    policy answers it with yaw.
    """
    terms = [
        ObsTerm("base_lin_vel", 3, "body frame"),
        ObsTerm("base_ang_vel", 3, "body frame"),
        ObsTerm("projected_gravity", 3, "body frame; pitch matters while digging"),
        ObsTerm("wheel_vel", num_wheels, "stall shows up here"),
        ObsTerm("arm_pos", num_arms, "boom angle"),
        ObsTerm("arm_vel", num_arms, ""),
        ObsTerm("drum_vel", num_arms, ""),
        ObsTerm("shroud_pos", num_arms, "inlet angle; the policy aims it"),
        ObsTerm("drum_phase", 2 * num_arms, "(sin, cos) of vanes * rotor angle"),
        ObsTerm("arm_torque", num_arms, "joint torque / ARM_EFFORT"),
        ObsTerm("drum_torque", num_arms, "joint torque / DRUM_EFFORT"),
        ObsTerm("drum_fill", num_arms, "estimated load / target_load_kg"),
        ObsTerm("target_level", 1, "drum height minus plane height, as the scan reads"),
        ObsTerm("grad_forward", 1, "commanded rise/run along body +x"),
        ObsTerm("grad_lateral", 1, "commanded rise/run along body +y"),
        ObsTerm("depth_error", 1, "ground minus target over the footprint; + is soil to cut"),
        ObsTerm("cut_progress", 1, "fraction of the footprint at or below target"),
        ObsTerm("terrain_scan", terrain_cells, "2-D, drum-centred heights"),
        ObsTerm("last_action", 5, "[forward, yaw, boom, drum, shroud]"),
    ]
    return ObsSpec(terms)


def critic_state_spec(terrain_cells: int = NAV_SCAN_CELLS, num_arms: int = 2) -> ObsSpec:
    """Privileged state for the critic only.

    The critic estimates value during training and is discarded afterwards, so
    it reads what no physical rover could: the true soil surface, exact
    captured mass, true slip. Isaac Lab plumbs it through `state_space`.
    """
    return ObsSpec([
        ObsTerm("terrain_scan_true", terrain_cells, "noise-free soil surface"),
        ObsTerm("drum_fill_mass", num_arms, "kg, exact"),
        ObsTerm("base_lin_vel_w", 3, "world frame"),
        ObsTerm("slip", num_arms, "commanded vs achieved contact speed"),
    ])
