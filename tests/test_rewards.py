"""Tests for the reward terms.

Each term is pinned on sign, on its value at the nominal state, and on the
direction it moves. A reward bug does not crash; it trains a confident policy
toward the wrong thing, so these are the only place the sign is ever checked.
"""

from __future__ import annotations

import math

import pytest
import torch

from luna_hifi_tasks.excavator.mdp.rewards import (
    TermLogger,
    action_rate_penalty,
    bearing_alignment,
    drift_penalty,
    drums_full,
    energy_penalty,
    fill_delta_reward,
    goal_reached,
    heading_error_sin_cos,
    idle_drum_penalty,
    nonfinite_state,
    progress_reward,
    stall_penalty,
    tipped,
    upright_penalty,
    wheel_slip,
    world_to_body_xy,
)

UPRIGHT = torch.tensor([[0.0, 0.0, -1.0]])


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------


def test_action_rate_is_zero_for_a_held_action_and_grows_with_change():
    a = torch.tensor([[0.5, -0.2]])
    assert action_rate_penalty(a, a).item() == 0.0
    assert action_rate_penalty(a, torch.zeros(1, 2)).item() == pytest.approx(0.29)
    assert (action_rate_penalty(torch.randn(8, 4), torch.randn(8, 4)) >= 0).all()


def test_energy_is_absolute_power():
    tau = torch.tensor([[10.0, -10.0]])
    vel = torch.tensor([[2.0, 2.0]])
    assert energy_penalty(tau, vel).item() == pytest.approx(40.0)  # never cancels


def test_upright_penalty_is_zero_upright_and_two_inverted():
    assert upright_penalty(UPRIGHT).item() == pytest.approx(0.0)
    assert upright_penalty(-UPRIGHT).item() == pytest.approx(2.0)
    tilted = torch.tensor([[0.0, math.sin(0.5), -math.cos(0.5)]])
    assert 0.0 < upright_penalty(tilted).item() < 1.0


def test_tipped_threshold_is_on_the_cosine():
    """A 60 degree tilt reads g_z = -0.5. Just past it must trip; just short
    must not."""
    cos60 = 0.5
    ok = torch.tensor([[0.0, 0.0, -0.55]])
    bad = torch.tensor([[0.0, 0.0, -0.45]])
    assert not tipped(ok, cos60).item()
    assert tipped(bad, cos60).item()
    assert not tipped(UPRIGHT, cos60).item()
    assert tipped(-UPRIGHT, cos60).item()


def test_nonfinite_flags_only_the_broken_env():
    pos = torch.zeros(3, 3)
    pos[1, 2] = float("nan")
    vel = torch.zeros(3, 3)
    vel[2, 0] = float("inf")
    got = nonfinite_state(pos, vel)
    assert got.tolist() == [False, True, True]


def test_wheel_slip_zero_when_rolling_and_positive_when_spinning():
    r = 0.30
    rolling = wheel_slip(torch.full((1, 4), 2.0), torch.tensor([0.6]), r)
    assert rolling.item() == pytest.approx(0.0, abs=1e-6)
    spinning = wheel_slip(torch.full((1, 4), 5.0), torch.tensor([0.0]), r)
    assert spinning.item() == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# navigation
# ---------------------------------------------------------------------------


def test_progress_is_positive_when_approaching():
    assert progress_reward(torch.tensor([5.0]), torch.tensor([4.5])).item() == pytest.approx(0.5)
    assert progress_reward(torch.tensor([5.0]), torch.tensor([5.5])).item() == pytest.approx(-0.5)


def test_bearing_alignment_is_plus_one_ahead_minus_one_behind_zero_at_goal():
    assert bearing_alignment(torch.tensor([[3.0, 0.0]])).item() == pytest.approx(1.0)
    assert bearing_alignment(torch.tensor([[-3.0, 0.0]])).item() == pytest.approx(-1.0)
    assert bearing_alignment(torch.tensor([[0.0, 2.0]])).item() == pytest.approx(0.0)
    assert bearing_alignment(torch.tensor([[0.0, 0.0]])).item() == 0.0


def test_goal_reached_needs_both_distance_and_heading():
    near, far = torch.tensor([0.2]), torch.tensor([2.0])
    facing, away = torch.tensor([0.99]), torch.tensor([0.0])
    assert goal_reached(near, facing, 0.5, 0.9).item()
    assert not goal_reached(far, facing, 0.5, 0.9).item()
    assert not goal_reached(near, away, 0.5, 0.9).item()


def test_heading_error_wraps_without_a_discontinuity():
    """A goal yaw of +179 deg and a base yaw of -179 deg differ by 2 deg, not
    358. sin/cos gets that right; a raw subtraction would not."""
    # base at -179 (= +181) and goal at +179: the goal is 2 degrees CLOCKWISE,
    # i.e. an error of -2, not +2 and certainly not 358.
    a = heading_error_sin_cos(torch.tensor([math.radians(179)]), torch.tensor([math.radians(-179)]))
    b = heading_error_sin_cos(torch.tensor([math.radians(-2)]), torch.tensor([0.0]))
    assert torch.allclose(a, b, atol=1e-6)
    assert a.shape == (1, 2)
    assert a[0, 0] < 0     # sin negative: turn right


def test_world_to_body_xy_rotates_by_yaw():
    v = torch.tensor([[1.0, 0.0]])
    # facing +y (yaw 90), a world +x vector is to the body's RIGHT (-y)
    got = world_to_body_xy(v, torch.tensor([math.pi / 2]))
    assert torch.allclose(got, torch.tensor([[0.0, -1.0]]), atol=1e-6)
    # facing +x, unchanged
    assert torch.allclose(world_to_body_xy(v, torch.tensor([0.0])), v, atol=1e-6)


# ---------------------------------------------------------------------------
# excavation
# ---------------------------------------------------------------------------


def test_fill_delta_sums_drums_and_goes_negative_on_spill():
    now = torch.tensor([[10.0, 12.0]])
    prev = torch.tensor([[8.0, 12.5]])
    assert fill_delta_reward(now, prev).item() == pytest.approx(1.5)
    assert fill_delta_reward(prev, now).item() == pytest.approx(-1.5)


def test_stall_is_zero_when_moving_as_commanded_and_positive_when_bogged():
    r = 0.30
    assert stall_penalty(torch.tensor([3.0]), torch.tensor([0.9]), r).item() == pytest.approx(0.0, abs=1e-6)
    assert stall_penalty(torch.tensor([3.0]), torch.tensor([0.0]), r).item() == pytest.approx(0.9)
    # moving faster than asked is not a stall
    assert stall_penalty(torch.tensor([1.0]), torch.tensor([2.0]), r).item() == 0.0


def test_drift_measures_uncommanded_motion_only():
    """The counter-rotation check: a machine moving exactly as commanded reads
    zero; a machine being shoved by its own dig reads the shove."""
    v = torch.tensor([[0.5, 0.0, 0.0]])
    assert drift_penalty(v, torch.tensor([0.5])).item() == pytest.approx(0.0)
    assert drift_penalty(v, torch.tensor([0.0])).item() == pytest.approx(0.5)
    sideways = torch.tensor([[0.0, 0.3, 0.0]])
    assert drift_penalty(sideways, torch.tensor([0.0])).item() == pytest.approx(0.3)


def test_idle_drum_penalises_spinning_only_when_nothing_is_captured():
    vel = torch.tensor([[6.0, -6.0]])
    assert idle_drum_penalty(vel, torch.tensor([0.0])).item() == pytest.approx(12.0)
    assert idle_drum_penalty(vel, torch.tensor([-0.1])).item() == pytest.approx(12.0)
    assert idle_drum_penalty(vel, torch.tensor([0.5])).item() == 0.0


def test_drums_full_requires_every_drum():
    assert drums_full(torch.tensor([[0.95, 0.96]]), 0.9).item()
    assert not drums_full(torch.tensor([[0.95, 0.40]]), 0.9).item()


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------


def test_term_logger_reports_per_second_means_and_resets():
    log = TermLogger(["a", "b"], num_envs=4, device="cpu")
    log.add("a", torch.tensor([1.0, 2.0, 3.0, 4.0]))
    log.add("a", torch.tensor([1.0, 2.0, 3.0, 4.0]))
    out = log.flush(torch.tensor([1, 3]), episode_len_s=2.0)
    assert out["Episode_Reward/a"] == pytest.approx((4.0 + 8.0) / 2 / 2.0)
    assert out["Episode_Reward/b"] == 0.0
    assert log.sums["a"].tolist() == [2.0, 0.0, 6.0, 0.0]
