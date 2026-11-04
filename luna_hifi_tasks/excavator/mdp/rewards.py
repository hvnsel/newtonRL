# Reward and termination terms as pure functions of tensors, each returning
# (num_envs,). "Penalty" functions return non-negative values; the env
# subtracts them with a weight. Body frame: +x forward, +y left, +z up;
# projected_gravity_b reads (0, 0, -1) upright.

from __future__ import annotations

import math

import torch


# --- shared ---


def action_rate_penalty(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """Sum of squared per-step action change."""
    return ((action - prev_action) ** 2).sum(dim=-1)


def energy_penalty(torque: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
    """Mechanical power, sum of |tau * omega| over joints. Both (E, J)."""
    return (torque * joint_vel).abs().sum(dim=-1)


def upright_penalty(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """Tilt in [0, 2]: 0 upright, 2 inverted."""
    return 1.0 + projected_gravity_b[:, 2]


def tipped(projected_gravity_b: torch.Tensor, max_tilt_cos: float) -> torch.Tensor:
    """Tilted past the angle whose cosine is max_tilt_cos."""
    return projected_gravity_b[:, 2] > -max_tilt_cos


def nonfinite_state(*tensors: torch.Tensor) -> torch.Tensor:
    """Envs with a non-finite entry in any of the (E, ...) tensors."""
    return torch.stack([~torch.isfinite(t.reshape(t.shape[0], -1)).all(dim=-1) for t in tensors]).any(dim=0)


def wheel_slip(wheel_vel: torch.Tensor, base_forward_speed: torch.Tensor, wheel_radius: float) -> torch.Tensor:
    """Mean |omega * r - v_x| over wheels, m/s. wheel_vel (E, W) rad/s."""
    return (wheel_vel * wheel_radius - base_forward_speed.unsqueeze(-1)).abs().mean(dim=-1)


# --- navigation ---


def progress_reward(dist_prev: torch.Tensor, dist_now: torch.Tensor) -> torch.Tensor:
    """Distance closed this step, m."""
    return dist_prev - dist_now


def bearing_alignment(goal_vec_b: torch.Tensor, forward_speed: torch.Tensor, speed_scale: float) -> torch.Tensor:
    """Cosine of the bearing to goal_vec_b times forward speed as a fraction
    of speed_scale: positive driving towards it with either end first."""
    dist = torch.linalg.norm(goal_vec_b, dim=-1)
    cos = torch.where(dist > 1e-3, goal_vec_b[:, 0] / dist.clamp_min(1e-3), torch.zeros_like(dist))
    return cos * (forward_speed / speed_scale).clamp(-1.0, 1.0)


def fade_in(x: torch.Tensor, start: float, full: float) -> torch.Tensor:
    """0 at or below start, 1 at or above full, linear between."""
    return ((x - start) / (full - start)).clamp(0.0, 1.0)


def goal_pose_ok(dist: torch.Tensor, heading_err: torch.Tensor, dist_tol: float, heading_tol: float) -> torch.Tensor:
    """Within dist_tol metres of the goal and heading_tol radians of its heading."""
    return (dist < dist_tol) & (heading_err.abs() < heading_tol)


def goal_reached(
    dist: torch.Tensor,
    heading_err: torch.Tensor,
    speed: torch.Tensor,
    yaw_rate: torch.Tensor,
    dist_tol: float,
    heading_tol: float,
    speed_tol: float,
    yaw_rate_tol: float,
) -> torch.Tensor:
    """goal_pose_ok, planar speed under speed_tol and |yaw rate| under
    yaw_rate_tol, on this step."""
    return goal_pose_ok(dist, heading_err, dist_tol, heading_tol) & (speed < speed_tol) & (yaw_rate.abs() < yaw_rate_tol)


def pose_potential(
    dist: torch.Tensor,
    heading_err: torch.Tensor,
    w_pos: float,
    pos_sigma: float,
    w_heading: float,
    heading_sigma: float,
    heading_radius: float,
) -> torch.Tensor:
    """Shaping potential for the final pose, <= 0 and 0 on the goal; the
    reward is its change over a step.

    -w_pos (1 - exp(-(d / pos_sigma)^2))
    - w_heading ramp (|psi| / pi + 1 - exp(-(psi / heading_sigma)^2)),
    ramp falling from 1 at the goal to 0 at heading_radius.
    """
    a = heading_err.abs()
    pos = 1.0 - torch.exp(-((dist / pos_sigma) ** 2))
    ramp = (1.0 - dist / heading_radius).clamp(0.0, 1.0)
    head = a / math.pi + 1.0 - torch.exp(-((a / heading_sigma) ** 2))
    return -(w_pos * pos + w_heading * ramp * head)


def settle_penalty(speed: torch.Tensor, yaw_rate: torch.Tensor, half_track: float) -> torch.Tensor:
    """Planar speed plus |yaw rate| times half the track, m/s."""
    return speed + half_track * yaw_rate.abs()


def axis_heading_sin_cos(goal_yaw: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
    """(sin, cos) of twice the yaw error, (E, 2): the same facing either way
    along the goal heading."""
    err = 2.0 * (goal_yaw - base_yaw)
    return torch.stack([torch.sin(err), torch.cos(err)], dim=-1)


def axis_heading_error(goal_yaw: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
    """Yaw error to the goal heading's axis, in (-pi/2, pi/2]. (E,)."""
    s = axis_heading_sin_cos(goal_yaw, base_yaw)
    return 0.5 * torch.atan2(s[:, 0], s[:, 1])


def body_to_world_xy(vec_b: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
    """Rotate (E, 2) body-frame xy into the world frame by base_yaw."""
    c, s = torch.cos(base_yaw), torch.sin(base_yaw)
    return torch.stack([c * vec_b[:, 0] - s * vec_b[:, 1], s * vec_b[:, 0] + c * vec_b[:, 1]], dim=-1)


def world_to_body_xy(vec_w: torch.Tensor, base_yaw: torch.Tensor) -> torch.Tensor:
    """Rotate (E, 2) world xy into the body yaw frame."""
    c, s = torch.cos(base_yaw), torch.sin(base_yaw)
    return torch.stack([c * vec_w[:, 0] + s * vec_w[:, 1], -s * vec_w[:, 0] + c * vec_w[:, 1]], dim=-1)


# --- excavation ---


def progress_delta(prev: torch.Tensor, now: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
    """prev - now, 0 where fresh marks an episode's first step."""
    return torch.where(fresh, torch.zeros_like(now), prev - now)


def spill_penalty(fill_delta: torch.Tensor) -> torch.Tensor:
    """Mass lost this step, non-negative."""
    return (-fill_delta).clamp_min(0.0)


def fill_delta_reward(fill_now: torch.Tensor, fill_prev: torch.Tensor) -> torch.Tensor:
    """Soil captured this step, summed over drums. (E, D) -> (E,)."""
    return (fill_now - fill_prev).sum(dim=-1)


def stall_penalty(wheel_cmd: torch.Tensor, base_forward_speed: torch.Tensor, wheel_radius: float) -> torch.Tensor:
    """Commanded surface speed (wheel_cmd rad/s) not turned into motion, m/s."""
    return ((wheel_cmd * wheel_radius).abs() - base_forward_speed.abs()).clamp_min(0.0)


def drift_penalty(base_lin_vel_b: torch.Tensor, forward_cmd_speed: torch.Tensor) -> torch.Tensor:
    """Uncommanded chassis motion, |v_x - v_cmd| + |v_y|, m/s."""
    return (base_lin_vel_b[:, 0] - forward_cmd_speed).abs() + base_lin_vel_b[:, 1].abs()


def drums_full(fill_fraction: torch.Tensor, threshold: float) -> torch.Tensor:
    """Every drum at or above threshold. (E, D) -> (E,)."""
    return (fill_fraction >= threshold).all(dim=-1)


def front_drum_full(fill_fraction: torch.Tensor, threshold: float) -> torch.Tensor:
    """Drum 0 at or above threshold. (E, D) -> (E,)."""
    return fill_fraction[:, 0] >= threshold


# --- bookkeeping ---


class TermLogger:
    """Sums each term's weighted contribution to the reward over the episode."""

    def __init__(self, names: list[str], num_envs: int, device) -> None:
        self.sums = {n: torch.zeros(num_envs, device=device) for n in names}

    def add_all(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        """Accumulate every term; return their sum, the step reward."""
        for name, value in terms.items():
            self.sums[name] += value
        return torch.stack(list(terms.values())).sum(dim=0)

    def flush(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Mean episode total per term over env_ids as Episode_Reward/<term>,
        then zero them."""
        out = {}
        n = max(int(env_ids.numel()), 1)
        for name, buf in self.sums.items():
            out[f"Episode_Reward/{name}"] = float(buf[env_ids].sum() / n)
            buf[env_ids] = 0.0
        return out
