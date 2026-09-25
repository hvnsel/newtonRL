# observations.py
#
# Observation assembly as a declared list of named terms.
#
#   * the observation dimension is derived from the terms
#   * every term declares its width and assembly asserts each tensor matches
#   * the spec hashes, so a policy trained against one layout cannot be loaded
#     against another
#
# Imports torch only, so the specs can be read without Isaac Lab.

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Scan geometry. The observation width is derived from it and the terrain
# sensor builds its patterns from it.
# ---------------------------------------------------------------------------

# Navigation, two windows: a coarse far one for routing and a fine near one
# for foot placement, mirroring a rover's mast camera and its hazcams.
#
# The far window is sized from stopping distance. With the arms stowed the
# front of the machine reaches x = 1.78 m, it halts in 1.10 m from 1.5 m/s,
# and a 16 x 8 grid at 0.45 m pushed 2.40 m forward ends at x = 5.775: four
# metres of ground ahead of the machine, 2.7 s at full speed.
NAV_FAR_NX, NAV_FAR_NY = 16, 8
NAV_FAR_CELL = 0.45
NAV_FAR_BIAS = 2.40
NAV_FAR_CELLS = NAV_FAR_NX * NAV_FAR_NY

# The near window is sized from the wheels: 12 x 8 at 0.20 m is 1.40 m wide
# against a machine 1.35 m over the tyres, one cell per wheel width.
NAV_NEAR_NX, NAV_NEAR_NY = 12, 8
NAV_NEAR_CELL = 0.20
NAV_NEAR_BIAS = 1.30
NAV_NEAR_CELLS = NAV_NEAR_NX * NAV_NEAR_NY

# Total navigation scan width, which is also what the critic gets clean.
NAV_SCAN_CELLS = NAV_FAR_CELLS + NAV_NEAR_CELLS

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
DIG_SCAN_SIZE = ((DIG_SCAN_NX - 1) * DIG_SCAN_CELL, (DIG_SCAN_NY - 1) * DIG_SCAN_CELL)


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
    """An ordered list of observation terms.

    Order is part of the contract the hash pins down: a policy trained with
    terms in one order cannot read an observation assembled in another.
    """

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
        """Stable 12-char digest of (name, dim) pairs in order.

        Ignores `note`, so reordering or resizing a term changes the hash and
        editing its comment does not.
        """
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


# ---------------------------------------------------------------------------
# The specs themselves
# ---------------------------------------------------------------------------
#
#   * every angle enters as (sin, cos), so nothing wraps at +-pi
#   * every height is relative, to the chassis or to the undisturbed bed
#     surface


def navigate_obs_spec(
    far_cells: int = NAV_FAR_CELLS,
    near_cells: int = NAV_NEAR_CELLS,
    num_arms: int = 2,
    num_wheels: int = 4,
) -> ObsSpec:
    """Observation for the navigation skill.

    The terrain scan carries the ground the rover has itself excavated: 0.19 m
    pits and spoil piles against a 0.30 m wheel radius, which proprioception
    reaches only once the machine is already on them.

    Two windows: far_scan reaches 4.0 m past the front of the machine for
    choosing a line, near_scan resolves one wheel width for choosing where to
    put a wheel.

    A cell the sensor did not measure reads scan_invalid, which sits outside
    the clipped range of a real height and so needs no separate mask.
    """
    terms = [
        ObsTerm("base_lin_vel", 3, "body frame"),
        ObsTerm("base_ang_vel", 3, "body frame"),
        ObsTerm("projected_gravity", 3, "body frame; also the tip-over signal"),
        ObsTerm("goal_vec_b", 2, "goal xy in BODY frame, scaled"),
        ObsTerm("goal_heading", 2, "(sin, cos) of heading error"),
        ObsTerm("wheel_vel", num_wheels, "joint velocities, scaled"),
        ObsTerm("arm_pos", num_arms, "held stowed here, but it moves the CG"),
        ObsTerm("drum_fill", num_arms, "0 on the rigid tier; keeps the layout"),
        ObsTerm("last_action", 2, "[forward, yaw]"),
        ObsTerm("far_scan", far_cells, "16x8 at 0.45 m, chassis-relative heights"),
        ObsTerm("near_scan", near_cells, "12x8 at 0.20 m, one cell per wheel width"),
    ]
    return ObsSpec(terms)


def excavate_obs_spec(
    terrain_cells: int = DIG_SCAN_CELLS,
    num_arms: int = 2,
    num_wheels: int = 4,
) -> ObsSpec:
    """Observation for the excavation skill.

    `terrain_scan` is a 2-D window, 2.0 x 1.0 m at 0.125 m, centred on the
    front drum. It carries the two things the machine answers laterally: the
    approach angle relative to the face, which it yaws to correct, and the
    lateral slope, which tips it.

    `drum_phase` is (sin, cos) of the vane count times the rotor angle, so it
    reads phase within a pocket and is identical for every pocket. Six vanes
    at 2.9 rad/s put a pocket mouth at the inlet every 0.36 s, 9 steps at
    25 Hz.

    `arm_torque` is the load sensor, and it weighs what the arm carries: soil
    the drum is buried in is held by the ground and does not appear here.

    The cut command is a plane, in three numbers. The rotor is a cylinder, so
    its cutting edge is a straight line across the swath and the depth is
    constant across it: the machine ramps along its direction of travel by
    raising the boom as it advances. `target_level` is in the same frame and
    units as every cell of terrain_scan, height relative to the front drum
    clipped to scan_clip, and the two gradients are dimensionless rise over
    run in the body frame.

    `grad_lateral` is the component the drum cannot cut away. The policy has
    yaw authority, so a lateral residual is the signal to turn.
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
