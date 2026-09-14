"""Tests for observation spec assembly.

The point of these is that every way of getting the observation layout wrong
raises instead of training badly: a missing term, a mistyped name, a term with
the wrong width, terms in the wrong order, or a checkpoint trained against a
different layout.
"""

from __future__ import annotations

import pytest
import torch

from luna_hifi_tasks.excavator.mdp.observations import (
    DIG_SCAN_CELL,
    DIG_SCAN_CELLS,
    DIG_SCAN_NX,
    NAV_SCAN_CELL,
    NAV_SCAN_CELLS,
    NAV_SCAN_NX,
    ObsSpec,
    ObsTerm,
    critic_state_spec,
    excavate_obs_spec,
    navigate_obs_spec,
)


def _parts(spec: ObsSpec, n: int = 5) -> dict[str, torch.Tensor]:
    return {t.name: torch.randn(n, t.dim) for t in spec.terms}


# ---------------------------------------------------------------------------
# basic structure
# ---------------------------------------------------------------------------


def test_dim_is_the_sum_of_terms():
    spec = ObsSpec([ObsTerm("a", 3), ObsTerm("b", 2), ObsTerm("c", 7)])
    assert spec.dim == 12


def test_zero_or_negative_dim_is_rejected():
    with pytest.raises(ValueError, match="non-positive"):
        ObsTerm("bad", 0)
    with pytest.raises(ValueError, match="non-positive"):
        ObsTerm("bad", -3)


def test_duplicate_term_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ObsSpec([ObsTerm("a", 3), ObsTerm("a", 4)])


def test_slice_of_locates_terms():
    spec = ObsSpec([ObsTerm("a", 3), ObsTerm("b", 2), ObsTerm("c", 7)])
    assert spec.slice_of("a") == slice(0, 3)
    assert spec.slice_of("b") == slice(3, 5)
    assert spec.slice_of("c") == slice(5, 12)
    with pytest.raises(KeyError):
        spec.slice_of("nope")


# ---------------------------------------------------------------------------
# assembly is checked
# ---------------------------------------------------------------------------


def test_assemble_concatenates_in_declared_order():
    spec = ObsSpec([ObsTerm("a", 2), ObsTerm("b", 3)])
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[3.0, 4.0, 5.0]])
    out = spec.assemble({"b": b, "a": a})          # dict order must not matter
    assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))
    assert torch.equal(out[:, spec.slice_of("a")], a)
    assert torch.equal(out[:, spec.slice_of("b")], b)


def test_assemble_rejects_a_missing_term():
    spec = navigate_obs_spec()
    parts = _parts(spec)
    parts.pop("goal_heading")
    with pytest.raises(KeyError, match="not provided"):
        spec.assemble(parts)


def test_assemble_rejects_an_unknown_term():
    spec = navigate_obs_spec()
    parts = _parts(spec)
    parts["goal_headnig"] = torch.zeros(5, 2)      # typo, the classic
    with pytest.raises(KeyError, match="not in the spec"):
        spec.assemble(parts)


def test_assemble_rejects_a_wrong_width():
    spec = navigate_obs_spec()
    parts = _parts(spec)
    parts["base_lin_vel"] = torch.zeros(5, 4)      # should be 3
    with pytest.raises(ValueError, match="declares dim 3 but got 4"):
        spec.assemble(parts)


def test_assemble_rejects_an_unbatched_tensor():
    spec = ObsSpec([ObsTerm("a", 3)])
    with pytest.raises(ValueError, match=r"must be \(num_envs, dim\)"):
        spec.assemble({"a": torch.zeros(3)})


def test_assemble_output_shape():
    spec = navigate_obs_spec()
    out = spec.assemble(_parts(spec, n=9))
    assert out.shape == (9, spec.dim)


# ---------------------------------------------------------------------------
# the schema hash
# ---------------------------------------------------------------------------


def test_hash_is_stable_across_calls_and_instances():
    assert navigate_obs_spec().schema_hash() == navigate_obs_spec().schema_hash()


def test_hash_changes_when_a_term_is_resized():
    a = ObsSpec([ObsTerm("x", 3), ObsTerm("y", 2)])
    b = ObsSpec([ObsTerm("x", 3), ObsTerm("y", 4)])
    assert a.schema_hash() != b.schema_hash()


def test_hash_changes_when_terms_are_reordered():
    """Order is part of the contract: same total width, different meaning."""
    a = ObsSpec([ObsTerm("x", 3), ObsTerm("y", 3)])
    b = ObsSpec([ObsTerm("y", 3), ObsTerm("x", 3)])
    assert a.dim == b.dim
    assert a.schema_hash() != b.schema_hash()


def test_hash_ignores_notes():
    """Editing a comment must not invalidate a checkpoint."""
    a = ObsSpec([ObsTerm("x", 3, "one wording")])
    b = ObsSpec([ObsTerm("x", 3, "another wording entirely")])
    assert a.schema_hash() == b.schema_hash()


def test_resizing_the_terrain_scan_changes_the_hash():
    """The whole reason the hash exists: a policy trained against one scan
    resolution must not load silently against another."""
    coarse = navigate_obs_spec(terrain_cells=192)
    fine = navigate_obs_spec(terrain_cells=384)
    assert coarse.schema_hash() != fine.schema_hash()
    assert fine.dim == coarse.dim + 192


# ---------------------------------------------------------------------------
# the concrete specs
# ---------------------------------------------------------------------------


def test_navigate_spec_always_carries_terrain():
    """Not optional. This rover drives over ground it has itself dug: 0.19 m
    pits against a 0.30 m wheel radius. Proprioception can react to a slope,
    never anticipate a pit."""
    spec = navigate_obs_spec()
    assert "terrain_scan" in spec.names
    assert spec.dim == 23 + NAV_SCAN_CELLS
    assert spec.dim == 215


def test_navigate_spec_carries_drum_fill_even_without_mpm():
    """Layout must not change between the rigid tier and the MPM tier, or a
    policy trained on one cannot be evaluated on the other."""
    assert "drum_fill" in navigate_obs_spec().names


def test_navigate_spec_has_no_raw_angle_terms():
    """Heading enters as (sin, cos). A raw wrapped angle is a discontinuity in
    the value function."""
    spec = navigate_obs_spec()
    assert spec.slice_of("goal_heading").stop - spec.slice_of("goal_heading").start == 2


def test_excavate_terrain_is_two_dimensional():
    """A 1-D strip along the approach would throw away both responses the
    machine actually has to lateral variation: it can yaw, and it can tip."""
    spec = excavate_obs_spec()
    sl = spec.slice_of("terrain_scan")
    assert sl.stop - sl.start == DIG_SCAN_CELLS == 128


def test_the_two_scans_are_sized_for_different_jobs():
    """Navigation needs reach, excavation needs resolution at the face."""
    assert NAV_SCAN_CELL > DIG_SCAN_CELL
    nav_window_x = NAV_SCAN_NX * NAV_SCAN_CELL
    dig_window_x = DIG_SCAN_NX * DIG_SCAN_CELL
    assert nav_window_x > dig_window_x
    assert nav_window_x >= 4.0        # 3.4 m machine + stopping distance


def test_excavate_action_width_matches_the_symmetric_action_space():
    spec = excavate_obs_spec()
    assert spec.slice_of("last_action").stop - spec.slice_of("last_action").start == 4


def test_critic_state_is_strictly_richer_than_the_actor_observation():
    """Asymmetric actor-critic: privileged soil state belongs to the critic."""
    actor = excavate_obs_spec()
    critic = critic_state_spec()
    assert "terrain_scan_true" in critic.names
    assert "terrain_scan_true" not in actor.names
    assert "drum_fill_mass" in critic.names          # exact kg
    assert "drum_fill" in actor.names                # normalised fraction only


def test_describe_renders_every_term():
    spec = navigate_obs_spec()
    text = spec.describe()
    for name in spec.names:
        assert name in text
    assert "hash=" in text
