# observations.py
#
# Observation assembly as a declared list of named terms rather than a
# hand-concatenated tensor and a magic `observation_space = 23`.
#
# Three things this buys, all of which are correctness rather than tidiness:
#
#   1. The observation dimension is DERIVED. Turning the height map on or off
#      is a flag, not an edit to a hardcoded integer that some other file also
#      hardcodes.
#   2. Assembly is checked. Every term declares its width, and building an
#      observation asserts each tensor matches. A term that silently returns
#      the wrong shape is otherwise invisible until the policy is mysteriously
#      worse.
#   3. The spec hashes. Store the hash with the checkpoint and a policy trained
#      against one layout cannot be silently loaded against another. In a
#      hierarchy this is the failure that costs you a week: the skill still
#      runs, still produces actions, and is just quietly wrong.
#
# No isaaclab import here on purpose -- same reason as sensors.py.

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


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
# Heights and angles follow two rules throughout:
#
#   * every angle enters as (sin, cos), never as a raw value. A heading error
#     wraps at +-pi, and that discontinuity quietly poisons the value function.
#   * every height is RELATIVE -- to the chassis, or to the undisturbed bed
#     surface. An absolute height makes the policy relearn the task at each
#     terrain elevation.


def navigate_obs_spec(
    height_map_cells: int = 0,
    num_arms: int = 2,
    num_wheels: int = 4,
) -> ObsSpec:
    """Observation for the navigation skill.

    `height_map_cells = 0` (the default) omits the terrain scan entirely.

    On a flat bed a height scan is a block of constant inputs, which is worse
    than useless: observation normalisation divides a constant channel by a
    near-zero standard deviation, and the policy spends capacity learning to
    ignore it. Turn it on when the terrain the rover drives over is terrain it
    has itself dug -- 0.19 m pits against a 0.30 m wheel radius is not
    something proprioception can anticipate, only react to.
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
    ]
    if height_map_cells:
        terms.append(ObsTerm("height_map", height_map_cells, "2-D scan, chassis-relative"))
    return ObsSpec(terms)


def excavate_obs_spec(
    profile_samples: int = 16,
    num_arms: int = 2,
    num_wheels: int = 4,
) -> ObsSpec:
    """Observation for the excavation skill.

    The terrain term here is a 1-D PROFILE along the approach direction, not a
    2-D grid, and that is a consequence of the machine's geometry rather than a
    shortcut. The drum is 1.00 m wide on a 1.35 m machine, so it cuts a
    full-width swath: there is no lateral degree of freedom for the policy to
    exercise, and a 2-D grid would spend ~200 inputs describing variation the
    drum cannot respond to differently. Sixteen samples along the direction of
    travel carry nearly the same information at a twelfth of the width.
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
        ObsTerm("soil_profile", profile_samples, "1-D strip along +x at the drum"),
        ObsTerm("last_action", 4, "[forward, yaw, boom, drum]"),
    ]
    return ObsSpec(terms)


def critic_state_spec(height_map_cells: int, num_arms: int = 2) -> ObsSpec:
    """Privileged state for the critic only.

    The critic estimates value during training and is thrown away afterwards,
    so it can see things no physical rover could: the true soil surface, exact
    captured mass, true slip. The actor must not, or the policy cannot transfer
    to a machine whose only terrain sense is a noisy, occluded depth camera.
    Isaac Lab plumbs this through `state_space`, which the tricycle left at 0.
    """
    return ObsSpec([
        ObsTerm("height_map_full", height_map_cells, "true 2-D soil surface"),
        ObsTerm("drum_fill_mass", num_arms, "kg, exact"),
        ObsTerm("base_lin_vel_w", 3, "world frame"),
        ObsTerm("slip", num_arms, "commanded vs achieved contact speed"),
    ])
