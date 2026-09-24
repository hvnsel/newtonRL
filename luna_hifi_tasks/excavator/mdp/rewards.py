# rewards.py
#
# Every reward and termination term as a pure function of tensors. Each
# returns (num_envs,) and nothing here reads an env object; the env sums them.
#
# Conventions:
#   * "penalty" functions return NON-NEGATIVE values and the env subtracts
#     them with a weight. A penalty that can go negative is a reward in
#     disguise and will be exploited.
#   * body frame: +x forward, +y left, +z up. projected_gravity_b is a unit
#     vector, so upright reads (0, 0, -1) at any gravity.

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def action_rate_penalty(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """Sum of squared per-step action change. Discourages the bang-bang
    policies that wreck contact-rich simulation and real actuators alike."""
    return ((action - prev_action) ** 2).sum(dim=-1)


def energy_penalty(torque: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
    """Mechanical power, sum of |tau * omega| over joints. Both (E, J)."""
    return (torque * joint_vel).abs().sum(dim=-1)


def upright_penalty(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """0 when upright, 2 when inverted. projected_gravity_b is a UNIT vector
    that reads (0, 0, -1) upright, so 1 + g_z is the tilt in [0, 2]."""
    return 1.0 + projected_gravity_b[:, 2]


def tipped(projected_gravity_b: torch.Tensor, max_tilt_cos: float) -> torch.Tensor:
    """True when the base has tilted past the angle whose cosine is
    `max_tilt_cos`. Upright g_z = -1; a 60 degree tilt reads -0.5."""
    return projected_gravity_b[:, 2] > -max_tilt_cos


def nonfinite_state(*tensors: torch.Tensor) -> torch.Tensor:
    """True for any env with a non-finite entry in any of the given (E, ...)
    tensors. Physics blow-ups terminate rather than poison the rollout."""
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

    Ignores yaw, so a turning skid-steer machine reads some slip by design:
    skid steering IS slipping the inside wheels. This term is a soft nudge
    against spinning, not a precise slip ratio."""
    surface = wheel_vel * wheel_radius
    return (surface - base_forward_speed.unsqueeze(-1)).abs().mean(dim=-1)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def progress_reward(dist_prev: torch.Tensor, dist_now: torch.Tensor) -> torch.Tensor:
    """Distance closed this step, metres. Positive when approaching. Dense,
    and the main thing the navigator is paid for."""
    return dist_prev - dist_now


def bearing_alignment(
    goal_vec_b: torch.Tensor,
    forward_speed: torch.Tensor,
    speed_scale: float,
) -> torch.Tensor:
    """Alignment with the goal, scaled by how fast the machine is closing on it.

    The cosine alone is a per-step payment for POINTING at the goal, and an
    episode that terminates on arrival forfeits the rest of it: standing still
    facing the goal for 500 steps outscored driving there. Multiplying by
    forward speed, clamped to [0, 1] of the machine's top speed, makes the
    term zero for a stationary machine and zero for one reversing away.
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
    """(sin, cos) of the yaw error, (E, 2). Never a raw angle: the wrap at
    +-pi is a discontinuity the value function cannot represent."""
    err = goal_yaw - base_yaw
    return torch.stack([torch.sin(err), torch.cos(err)], dim=-1)


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
    first step. Without the mask the opening step scores the whole quantity as
    if the policy had just created it."""
    return torch.where(fresh, torch.zeros_like(now), prev - now)


def spill_penalty(fill_delta: torch.Tensor) -> torch.Tensor:
    """Mass lost this step, non-negative. (E,) -> (E,).

    The fill term already scores a loss negatively; this one is the extra
    asymmetry, so shedding a kilogram costs more than capturing one pays."""
    return (-fill_delta).clamp_min(0.0)


def fill_delta_reward(fill_now: torch.Tensor, fill_prev: torch.Tensor) -> torch.Tensor:
    """Soil captured this step, summed over drums. (E, D) -> (E,). The primary
    excavation signal. Negative when soil is spilled, which is exactly right."""
    return (fill_now - fill_prev).sum(dim=-1)


def stall_penalty(
    wheel_cmd: torch.Tensor,           # (E,) commanded forward wheel speed, rad/s
    base_forward_speed: torch.Tensor,  # (E,)
    wheel_radius: float,
) -> torch.Tensor:
    """Commanded surface speed the machine failed to turn into motion, m/s.
    Zero when it moves as fast as the wheels ask; large when it is bogged."""
    asked = (wheel_cmd * wheel_radius).abs()
    got = base_forward_speed.abs()
    return (asked - got).clamp_min(0.0)


def drift_penalty(
    base_lin_vel_b: torch.Tensor,      # (E, 3)
    forward_cmd_speed: torch.Tensor,   # (E,) m/s the policy asked for
) -> torch.Tensor:
    """Motion of the chassis the policy did not command, m/s.

    On the Moon this is the counter-rotation check. Available traction is ~782
    N for the whole machine and a drum cutting cohesive regolith can exceed
    that, so a machine whose drum reactions do not cancel gets shoved. This
    term is that shove, measured directly: |v_x - v_cmd| + |v_y|.
    """
    return (base_lin_vel_b[:, 0] - forward_cmd_speed).abs() + base_lin_vel_b[:, 1].abs()


def drums_full(fill_fraction: torch.Tensor, threshold: float) -> torch.Tensor:
    """(E, D) -> (E,). True when EVERY drum has reached the threshold."""
    return (fill_fraction >= threshold).all(dim=-1)


def front_drum_full(fill_fraction: torch.Tensor, threshold: float) -> torch.Tensor:
    """(E, D) -> (E,). True when the FRONT drum has reached the threshold.
    Driving forward the front drum is the one meeting fresh soil; the rear
    trails through the trench and in a pile-in-front scene never reaches it."""
    return fill_fraction[:, 0] >= threshold


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------


class TermLogger:
    """Accumulates per-term reward sums so training logs show which term is
    actually driving the policy. Without this you learn that something is
    wrong from the behaviour, weeks later."""

    def __init__(self, names: list[str], num_envs: int, device) -> None:
        self.sums = {n: torch.zeros(num_envs, device=device) for n in names}

    def add(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self.sums[name] += value
        return value

    def flush(self, env_ids: torch.Tensor, episode_len_s: float) -> dict[str, float]:
        """Mean per-second contribution over the finishing envs, then zero them."""
        out = {}
        n = max(int(env_ids.numel()), 1)
        for name, buf in self.sums.items():
            out[f"Episode_Reward/{name}"] = float(buf[env_ids].sum() / n / episode_len_s)
            buf[env_ids] = 0.0
        return out
