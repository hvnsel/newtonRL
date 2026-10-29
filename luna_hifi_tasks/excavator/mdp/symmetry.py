# symmetry.py
#
# The navigate machine turned 180 degrees about its vertical axis, as rsl_rl's
# symmetry augmentation function (RslRlSymmetryCfg.data_augmentation_func).
#
# The rear assembly is the front one turned 180 degrees (excavator.py builds
# it with yaw = pi), so the machine facing one way is the same machine facing
# the other with its ends relabelled, and a policy driving it should do the
# same thing either way round. rsl_rl doubles every PPO mini-batch with each
# sample turned round, so the actor and the critic learn both at once and
# neither end becomes the front.
#
# Turning about z negates body x and y and keeps z; world-frame terms keep
# all three. The wheels trade places diagonally and spin the other way about
# an axle that now points the other way. The front and rear assemblies trade
# places. The terrain windows are centred on the chassis, so each maps onto
# itself turned 180 degrees: both image axes flipped, or in the row-major
# vector, the vector reversed. Every reward term and termination reads the
# same for the turned machine, which is what makes the doubled samples true
# ones: the goal heading is an axis and bearing pays speed towards the goal at
# either end.

from __future__ import annotations

import torch

from ..excavator import JOINT_ARMS, JOINT_WHEELS
from .observations import NAV_FAR_CELLS, NAV_SCAN_CELLS, critic_state_spec, navigate_obs_spec

# (forward, yaw): driving forward turned round is driving backward; a turn
# about z is the same turn.
ACTION_SIGN = (-1.0, 1.0)


def _turned_wheel(name: str) -> str:
    """The wheel that takes `name`'s place: front <-> rear and left <-> right,
    so wheel_fl <-> wheel_rr and wheel_rl <-> wheel_fr."""
    end, side = name[-2], name[-1]
    return name[:-2] + {"f": "r", "r": "f"}[end] + {"l": "r", "r": "l"}[side]


def _turned_assembly(name: str) -> str:
    return name.replace("front", "@").replace("rear", "front").replace("@", "rear")


def _order(names: list[str], turned) -> tuple[int, ...]:
    """Where each slot reads from once turned: slot i takes the value of the
    joint that turns into names[i]."""
    return tuple(names.index(turned(n)) for n in names)


_XY = ((0, 1, 2), (-1.0, -1.0, 1.0))
_WHEELS = (_order(JOINT_WHEELS, _turned_wheel), (-1.0,) * len(JOINT_WHEELS))
_ENDS = (_order(JOINT_ARMS, _turned_assembly), (1.0,) * len(JOINT_ARMS))

# Per policy term: (source order, sign). Terms not listed are unchanged:
# goal_heading is (sin, cos) of twice the heading error, which a half turn
# leaves alone.
_POLICY_RULES = {
    "base_lin_vel": _XY,
    "base_ang_vel": _XY,
    "projected_gravity": _XY,
    "goal_vec_b": ((0, 1), (-1.0, -1.0)),
    "goal_vec_near": ((0, 1), (-1.0, -1.0)),
    "wheel_vel": _WHEELS,
    "arm_pos": _ENDS,
    "drum_fill": _ENDS,
    "last_action": ((0, 1), ACTION_SIGN),
}
# The critic's own group. base_lin_vel_w is world frame and unchanged. slip is
# (left, right), each side's mean |omega r - v|, which the turned machine reads
# off the other side.
_CRITIC_RULES = {
    "terrain_scan_true": (
        tuple(range(NAV_FAR_CELLS - 1, -1, -1)) + tuple(range(NAV_SCAN_CELLS - 1, NAV_FAR_CELLS - 1, -1)),
        (1.0,) * NAV_SCAN_CELLS,
    ),
    "drum_fill_mass": _ENDS,
    "slip": ((1, 0), (1.0, 1.0)),
}


def _signed_permutation(spec, rules: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(source index, sign) per slot of the assembled vector."""
    src = torch.arange(spec.dim)
    sign = torch.ones(spec.dim)
    for name, (order, signs) in rules.items():
        s = spec.slice_of(name)
        if len(order) != s.stop - s.start or len(signs) != len(order):
            raise ValueError(f"symmetry rule for {name!r} covers {len(order)} of {s.stop - s.start} slots")
        src[s] = s.start + torch.tensor(order)
        sign[s] = torch.tensor(signs)
    return src, sign


_VECTOR_MAPS = {
    "policy": _signed_permutation(navigate_obs_spec(), _POLICY_RULES),
    "critic": _signed_permutation(critic_state_spec(terrain_cells=NAV_SCAN_CELLS), _CRITIC_RULES),
}
_IMAGES = ("far_scan", "near_scan")
_on_device: dict = {}


def _vector_map(key: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    hit = _on_device.get((key, device))
    if hit is None:
        src, sign = _VECTOR_MAPS[key]
        hit = _on_device[(key, device)] = (src.to(device), sign.to(device))
    return hit


def turn_group(key: str, x: torch.Tensor) -> torch.Tensor:
    """One observation group of a batch, as the machine turned 180 degrees."""
    if key in _IMAGES:
        return torch.flip(x, dims=(-2, -1))
    if key in _VECTOR_MAPS:
        src, sign = _vector_map(key, x.device)
        return x[:, src] * sign
    raise KeyError(f"no 180-degree rule for observation group {key!r}; add one to mdp/symmetry.py")


def turn_actions(actions: torch.Tensor) -> torch.Tensor:
    return actions * torch.tensor(ACTION_SIGN, device=actions.device, dtype=actions.dtype)


@torch.no_grad()
def turn_around(env=None, obs=None, actions=None):
    """rsl_rl's symmetry hook. Returns (obs, actions), each twice the batch:
    the samples as given, then the same samples turned 180 degrees. Either is
    None when its input is."""
    obs_aug = None
    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for key in obs.keys():
            obs_aug[key][n:] = turn_group(key, obs[key])
    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat([actions, turn_actions(actions)], dim=0)
    return obs_aug, actions_aug
