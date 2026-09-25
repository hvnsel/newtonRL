# rewards.py
#
# Every reward and termination term as a pure function of tensors. Each
# returns (num_envs,) and nothing here reads an env object; the env sums them.
#
# Conventions:
#   * "penalty" functions return non-negative values and the env subtracts
#     them with a weight.
#   * body frame: +x forward, +y left, +z up. projected_gravity_b is a unit
#     vector, so upright reads (0, 0, -1) at any gravity.

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def action_rate_penalty(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """Sum of squared per-step action change."""
    return ((action - prev_action) ** 2).sum(dim=-1)


def energy_penalty(torque: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
    """Mechanical power, sum of |tau * omega| over joints. Both (E, J)."""
    return (torque * joint_vel).abs().sum(dim=-1)


def upright_penalty(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """0 when upright, 2 when inverted. projected_gravity_b is a unit vector
    reading (0, 0, -1) upright, so 1 + g_z is the tilt in [0, 2]."""
    return 1.0 + projected_gravity_b[:, 2]


def tipped(projected_gravity_b: torch.Tensor, max_tilt_cos: float) -> torch.Tensor:
    """True when the base has tilted past the angle whose cosine is
    `max_tilt_cos`. Upright g_z = -1; a 60 degree tilt reads -0.5."""
    return projected_gravity_b[:, 2] > -max_tilt_cos


def nonfinite_state(*tensors: torch.Tensor) -> torch.Tensor:
    """True for any env with a non-finite entry in any of the given (E, ...)
    tensors. The env terminates on it."""
    bad = None
    for t in tensors:
        flat = t.reshape(t.shape[0], -1)
        b = ~torch.isfinite(flat).all(dim=-1)
        bad = b if bad is None else (bad | b)
    return bad


def wheel_slip(
    wheel_vel: torch.Tensor,          # (E, W) joint rad/s
    base_forward_speed: torch.Tensor, # (E,) body-frame x
    wheel_radius: float,
) -> torch.Tensor:
    """Mean |omega*r - v| across wheels, m/s. Zero when rolling cleanly.

    Ignores yaw, so a turning skid-steer machine reads some slip: skid
    steering slips the inside wheels."""
    surface = wheel_vel * wheel_radius
    return (surface - base_forward_speed.unsqueeze(-1)).abs().mean(dim=-1)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def progress_reward(dist_prev: torch.Tensor, dist_now: torch.Tensor) -> torch.Tensor:
    """Distance closed this step, metres. Positive when approaching."""
    return dist_prev - dist_now


def bearing_alignment(
    goal_vec_b: torch.Tensor,
    forward_speed: torch.Tensor,
    speed_scale: float,
) -> torch.Tensor:
    """Alignment with the goal, scaled by how fast the machine is closing on it.

    The cosine of the bearing times forward speed, clamped to [0, 1] of the
    machine's top speed, so the term is zero for a stationary machine and for
    one reversing away.
    """
    dist = torch.linalg.norm(goal_vec_b, dim=-1)
    cos = torch.where(dist > 1e-3, goal_vec_b[:, 0] / dist.clamp_min(1e-3), torch.zeros_like(dist))
    return cos * (forward_speed / speed_scale).clamp(0.0, 1.0)


def goal_reached(
    dist: torch.Tensor,
    heading_err_cos: torch.Tensor,
    dist_tol: float,
    heading_tol_cos: float,
) -> torch.Tensor:
    """Within `dist_tol` metres and facing within the tolerance angle."""
    return (dist < dist_tol) & (heading_err_cos > heading_tol_cos)


def heading_error_sin_cos(goal_yaw: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
    """(sin, cos) of the yaw error, (E, 2), so nothing wraps at +-pi."""
    err = goal_yaw - base_yaw
    return torch.stack([torch.sin(err), torch.cos(err)], dim=-1)


def body_to_world_xy(vec_b: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
    """Rotate a body-frame xy vector out into the world yaw frame. The inverse
    of world_to_body_xy. (E, 2) -> (E, 2)."""
    c, s = torch.cos(base_yaw), torch.sin(base_yaw)
    x = c * vec_b[:, 0] - s * vec_b[:, 1]
    y = s * vec_b[:, 0] + c * vec_b[:, 1]
    return torch.stack([x, y], dim=-1)


def world_to_body_xy(vec_w: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
    """Rotate a world-frame xy vector into the yaw frame. (E, 2) -> (E, 2)."""
    c, s = torch.cos(base_yaw), torch.sin(base_yaw)
    x = c * vec_w[:, 0] + s * vec_w[:, 1]
    y = -s * vec_w[:, 0] + c * vec_w[:, 1]
    return torch.stack([x, y], dim=-1)


# ---------------------------------------------------------------------------
# Excavation
# ---------------------------------------------------------------------------


def progress_delta(
    prev: torch.Tensor,
    now: torch.Tensor,
    fresh: torch.Tensor,
) -> torch.Tensor:
    """Reduction from `prev` to `now`, zero where `fresh` marks an episode's
    first step, which has no previous value."""
    return torch.where(fresh, torch.zeros_like(now), prev - now)


def spill_penalty(fill_delta: torch.Tensor) -> torch.Tensor:
    """Mass lost this step, non-negative. (E,) -> (E,).

    Paid on top of the negative fill term, so shedding a kilogram costs more
    than capturing one pays."""
    return (-fill_delta).clamp_min(0.0)


def fill_delta_reward(fill_now: torch.Tensor, fill_prev: torch.Tensor) -> torch.Tensor:
    """Soil captured this step, summed over drums. (E, D) -> (E,). Negative
    when soil is spilled."""
    return (fill_now - fill_prev).sum(dim=-1)


def stall_penalty(
    wheel_cmd: torch.Tensor,           # (E,) commanded forward wheel speed, rad/s
    base_forward_speed: torch.Tensor,  # (E,)
    wheel_radius: float,
) -> torch.Tensor:
    """Commanded surface speed not turned into motion, m/s. Zero when the
    machine moves as fast as the wheels ask, large when it is bogged."""
    asked = (wheel_cmd * wheel_radius).abs()
    got = base_forward_speed.abs()
    return (asked - got).clamp_min(0.0)


def drift_penalty(
    base_lin_vel_b: torch.Tensor,      # (E, 3)
    forward_cmd_speed: torch.Tensor,   # (E,) m/s the policy asked for
) -> torch.Tensor:
    """Motion of the chassis the policy did not command, m/s:
    |v_x - v_cmd| + |v_y|.

    The counter-rotation check. Available traction is ~782 N for the whole
    machine and a drum cutting cohesive regolith can exceed that, so drum
    reactions that do not cancel show up here as a shove.
    """
    return (base_lin_vel_b[:, 0] - forward_cmd_speed).abs() + base_lin_vel_b[:, 1].abs()


def drums_full(fill_fraction: torch.Tensor, threshold: float) -> torch.Tensor:
    """(E, D) -> (E,). True when every drum has reached the threshold."""
    return (fill_fraction >= threshold).all(dim=-1)


def front_drum_full(fill_fraction: torch.Tensor, threshold: float) -> torch.Tensor:
    """(E, D) -> (E,). True when the front drum has reached the threshold.
    Driving forward, the front drum is the one meeting fresh soil."""
    return fill_fraction[:, 0] >= threshold


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


class TermLogger:
    """Accumulates each term's signed, weighted contribution to the reward,
    which is what the policy optimises.

    Totals are summed over the episode, so an episode that ends early reads as
    having paid less.
    """

    def __init__(self, names: list[str], num_envs: int, device) -> None:
        self.names = list(names)
        self.sums = {n: torch.zeros(num_envs, device=device) for n in names}

    def add_all(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        """Accumulate every term and return their sum, the step reward.

        Raises on a term the logger was not built with, or a declared term
        this step left out.
        """
        missing = [n for n in self.names if n not in terms]
        extra = sorted(set(terms) - set(self.names))
        if missing or extra:
            raise KeyError(f"reward terms missing={missing} unexpected={extra}")
        total = None
        for name in self.names:
            value = terms[name]
            self.sums[name] += value
            total = value if total is None else total + value
        return total

    def flush(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Mean episode total per term over the finishing envs, then zero
        them. The values sum to the mean episode return."""
        out = {}
        n = max(int(env_ids.numel()), 1)
        for name, buf in self.sums.items():
            out[f"Episode_Reward/{name}"] = float(buf[env_ids].sum() / n)
            buf[env_ids] = 0.0
        return out
