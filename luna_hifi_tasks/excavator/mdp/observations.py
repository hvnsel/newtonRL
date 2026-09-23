# observations.py
#
# Observation assembly as a declared list of named terms.
#
#   * the observation dimension is derived, so turning the height map on or
#     off is a flag rather than an integer edited in two files
#   * every term declares its width and assembly asserts each tensor matches
#   * the spec hashes, so a policy trained against one layout cannot be loaded
#     against another
#
# No isaaclab import here, for the same reason as sensors.py.

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Scan geometry. Declared here because the observation width depends on it, and
# imported by the terrain sensor so the two can never disagree.
# ---------------------------------------------------------------------------

# Navigation: far enough ahead to stop and turn, coarse enough that every cell
# is a feature the machine can actually act on.
NAV_SCAN_NX, NAV_SCAN_NY = 16, 12
NAV_SCAN_CELL = 0.25
NAV_SCAN_FORWARD_BIAS = 0.5     # metres the window is pushed ahead of the chassis
NAV_SCAN_CELLS = NAV_SCAN_NX * NAV_SCAN_NY

# Excavation: the cutting face, at the FRONT drum. With symmetric arm control
# the policy cannot act differently on the two ends, and driving forward the
# front drum is the one meeting fresh soil, so one patch there is the whole
# usable picture. Independent arm control would want a second patch.
DIG_SCAN_NX, DIG_SCAN_NY = 16, 8
DIG_SCAN_CELL = 0.125
DIG_SCAN_CELLS = DIG_SCAN_NX * DIG_SCAN_NY

# GridPatternCfg puts a point at both ends of each axis:
# arange(-size/2, size/2 + eps, res) yields size/res + 1 points. For exactly
# NX x NY rays the pattern size must be (NX-1)*cell, not NX*cell, which would
# give 17 x 13 = 221 rays against a 192-wide term.
NAV_SCAN_SIZE = ((NAV_SCAN_NX - 1) * NAV_SCAN_CELL, (NAV_SCAN_NY - 1) * NAV_SCAN_CELL)
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

    Order is part of the contract: a policy trained with terms in one order
    cannot read an observation assembled in another. That is exactly what the
    hash pins down.
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

        Deliberately ignores `note`: editing a comment must not invalidate a
        checkpoint, but reordering or resizing terms must.
        """
        payload = "|".join(f"{t.name}:{t.dim}" for t in self.terms)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def slice_of(self, name: str) -> slice:
        """Where a term sits in the assembled vector. For debugging and for
        logging per-term statistics when an observation goes non-finite."""
        off = 0
        for t in self.terms:
            if t.name == name:
                return slice(off, off + t.dim)
            off += t.dim
        raise KeyError(f"no observation term named {name!r}; have {self.names}")

    def assemble(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatenate terms in declared order, checking every shape.

        Raises on a missing term, an extra term, or a width mismatch. All three
        are bugs that otherwise surface as a policy that trains badly for
        reasons nobody can find.
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
#   * every angle enters as (sin, cos): a raw heading error wraps at +-pi and
#     the discontinuity poisons the value function
#   * every height is relative, to the chassis or to the undisturbed bed
#     surface, so the policy does not relearn the task per terrain elevation


def navigate_obs_spec(
    terrain_cells: int = NAV_SCAN_CELLS,
    num_arms: int = 2,
    num_wheels: int = 4,
) -> ObsSpec:
    """Observation for the navigation skill.

    The terrain scan is NOT optional. This rover drives over ground it has
    itself excavated: 0.19 m pits and spoil piles against a 0.30 m wheel
    radius. Proprioception lets a policy react to a slope it is already on; it
    can never anticipate a pit. A blind navigator is not a simpler version of
    this skill, it is a different and worse one.

    Window is 4.0 x 3.0 m at 0.25 m, chassis-centred and slightly forward-
    biased. That is sized from stopping distance: ~1.1 m to halt from 1.5 m/s,
    on a machine that is itself 3.4 m long. Cell size is set by the features
    that matter -- a 1.0 m drum cuts a 1.0 m swath, so sub-0.25 m detail is
    below what the machine can act on.
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
        ObsTerm("terrain_scan", terrain_cells, "2-D, chassis-relative heights"),
    ]
    return ObsSpec(terms)


def excavate_obs_spec(
    terrain_cells: int = DIG_SCAN_CELLS,
    num_arms: int = 2,
    num_wheels: int = 4,
) -> ObsSpec:
    """Observation for the excavation skill.

    A 2-D scan, not a 1-D profile along the approach direction. The full-width
    drum is a tempting argument for 1-D -- it cuts the whole swath at once, so
    there is no lateral choice about WHERE in the cut to bite -- but the
    machine still has two responses to lateral variation that a strip would
    throw away: it can yaw, so approach angle relative to the face is a real
    decision, and it can tip, so lateral slope is a safety input. 128 cells
    against 16 is a cheap price for both.

    Window is 2.0 x 1.0 m at 0.125 m, centred on the ACTIVE drum rather than
    the chassis. Finer and smaller than the navigation scan because the
    quantity of interest is the shape of the cutting face, not the route.
    """
    terms = [
        ObsTerm("base_lin_vel", 3, "body frame"),
        ObsTerm("base_ang_vel", 3, "body frame"),
        ObsTerm("projected_gravity", 3, "body frame; pitch matters while digging"),
        ObsTerm("wheel_vel", num_wheels, "stall shows up here"),
        ObsTerm("arm_pos", num_arms, "boom angle"),
        ObsTerm("arm_vel", num_arms, ""),
        ObsTerm("drum_vel", num_arms, ""),
        ObsTerm("drum_fill", num_arms, "fraction of bore capacity"),
        ObsTerm("terrain_scan", terrain_cells, "2-D, drum-centred heights"),
        ObsTerm("last_action", 4, "[forward, yaw, boom, drum]"),
    ]
    return ObsSpec(terms)


def critic_state_spec(terrain_cells: int = NAV_SCAN_CELLS, num_arms: int = 2) -> ObsSpec:
    """Privileged state for the critic only.

    The critic estimates value during training and is thrown away afterwards,
    so it can see things no physical rover could: the true soil surface, exact
    captured mass, true slip. The actor must not, or the policy cannot transfer
    to a machine whose only terrain sense is a noisy, occluded depth camera.
    Isaac Lab plumbs this through `state_space`, which the tricycle left at 0.
    """
    return ObsSpec([
        ObsTerm("terrain_scan_true", terrain_cells, "noise-free soil surface"),
        ObsTerm("drum_fill_mass", num_arms, "kg, exact"),
        ObsTerm("base_lin_vel_w", 3, "world frame"),
        ObsTerm("slip", num_arms, "commanded vs achieved contact speed"),
    ])
