"""Tests for the terrain scan.

The property that matters most is the last one in this file: the two backends
must return the same numbers for the same ground. If they diverge, a skill
trained on procedural terrain silently misreads MPM soil, and the fidelity
tier stops being a config choice.
"""

from __future__ import annotations

import math

import pytest
import torch

from luna_hifi_tasks.excavator.mdp.observations import (
    DIG_SCAN_CELLS,
    NAV_SCAN_CELL,
    NAV_SCAN_CELLS,
    NAV_SCAN_NX,
    NAV_SCAN_NY,
)
from luna_hifi_tasks.excavator.mdp.terrain import (
    dig_scan_pattern,
    generate_excavation_terrain,
    nav_scan_pattern,
    sample_height_grid,
    scan_from_heightfield,
    scan_from_particles,
    scan_pattern,
    scan_points_world,
    yaw_from_quat,
)

IDENT = torch.tensor([[0.0, 0.0, 0.0, 1.0]])   # xyzw, Isaac Lab 3.x


def _quat_z(a: float) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, math.sin(a / 2), math.cos(a / 2)]])


def _quat_y(a: float) -> torch.Tensor:
    return torch.tensor([[0.0, math.sin(a / 2), 0.0, math.cos(a / 2)]])


# ---------------------------------------------------------------------------
# pattern
# ---------------------------------------------------------------------------


def test_pattern_shape_and_counts():
    p = nav_scan_pattern()
    assert p.shape == (NAV_SCAN_CELLS, 2)
    assert dig_scan_pattern().shape == (DIG_SCAN_CELLS, 2)


def test_pattern_is_centred_apart_from_the_forward_bias():
    p = scan_pattern(4, 4, 0.5, forward_bias=0.0)
    assert p[:, 0].mean().abs() < 1e-6
    assert p[:, 1].mean().abs() < 1e-6
    biased = scan_pattern(4, 4, 0.5, forward_bias=1.25)
    assert biased[:, 0].mean() == pytest.approx(1.25)
    assert biased[:, 1].mean().abs() < 1e-6


def test_pattern_spacing_matches_cell_size():
    p = scan_pattern(5, 3, 0.25).reshape(3, 5, 2)
    assert (p[0, 1, 0] - p[0, 0, 0]) == pytest.approx(0.25)
    assert (p[1, 0, 1] - p[0, 0, 1]) == pytest.approx(0.25)


def test_nav_window_covers_the_machine_plus_stopping_distance():
    assert NAV_SCAN_NX * NAV_SCAN_CELL >= 4.0
    assert NAV_SCAN_NY * NAV_SCAN_CELL >= 3.0


# ---------------------------------------------------------------------------
# placing the pattern in the world
# ---------------------------------------------------------------------------


def test_yaw_from_quat():
    for a in (0.0, 0.7, -2.1, 3.0):
        assert yaw_from_quat(_quat_z(a)).item() == pytest.approx(a, abs=1e-5)


def test_scan_points_translate_with_the_base():
    p = scan_pattern(3, 3, 0.5)
    at_origin = scan_points_world(torch.zeros(1, 3), IDENT, p)
    shifted = scan_points_world(torch.tensor([[7.0, -2.0, 5.0]]), IDENT, p)
    assert torch.allclose(shifted - at_origin, torch.tensor([7.0, -2.0]).expand(1, 9, 2))


def test_scan_points_rotate_with_yaw():
    p = scan_pattern(3, 3, 0.5)
    rot = scan_points_world(torch.zeros(1, 3), _quat_z(math.pi / 2), p)
    # a 90 deg yaw maps local +x to world +y
    base = scan_points_world(torch.zeros(1, 3), IDENT, p)
    assert torch.allclose(rot[..., 0], -base[..., 1], atol=1e-5)
    assert torch.allclose(rot[..., 1], base[..., 0], atol=1e-5)


def test_scan_points_ignore_pitch_and_roll():
    """The grid must not tilt with chassis attitude. Otherwise identical
    terrain reads differently depending on how the machine is sitting, and the
    policy has to learn to undo its own suspension geometry."""
    p = scan_pattern(5, 5, 0.3)
    flat = scan_points_world(torch.zeros(1, 3), IDENT, p)
    for pitch in (0.2, -0.45, 0.8):
        tilted = scan_points_world(torch.zeros(1, 3), _quat_y(pitch), p)
        assert torch.allclose(flat, tilted, atol=1e-5), pitch


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def test_sample_recovers_a_constant_field():
    grid = torch.full((1, 8, 8), 0.37)
    q = torch.tensor([[[0.1, 0.1], [0.3, -0.2]]])
    got = sample_height_grid(grid, q, torch.zeros(1, 3), (-0.4, -0.4), 0.1, outside_value=-9.0)
    assert torch.allclose(got, torch.full_like(got, 0.37), atol=1e-6)


def test_sample_interpolates_a_linear_ramp():
    """A plane must be reproduced exactly by bilinear interpolation."""
    nx = ny = 10
    cell = 0.2
    lower = (-1.0, -1.0)
    xs = (torch.arange(nx, dtype=torch.float32) + 0.5) * cell + lower[0]
    ys = (torch.arange(ny, dtype=torch.float32) + 0.5) * cell + lower[1]
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = (0.5 * gx + 0.25 * gy).unsqueeze(0)

    q = torch.tensor([[[0.13, -0.07], [-0.4, 0.31], [0.0, 0.0]]])
    got = sample_height_grid(grid, q, torch.zeros(1, 3), lower, cell, outside_value=-9.0)
    want = 0.5 * q[..., 0] + 0.25 * q[..., 1]
    assert torch.allclose(got, want, atol=1e-5), (got, want)


def test_sample_returns_outside_value_beyond_the_grid():
    """Not the clamped edge height: clamping invents a flat plateau extending
    forever past the bed, which a policy will learn to drive onto."""
    grid = torch.full((1, 6, 6), 1.0)
    q = torch.tensor([[[50.0, 0.0], [0.0, -50.0], [0.05, 0.05]]])
    got = sample_height_grid(grid, q, torch.zeros(1, 3), (-0.3, -0.3), 0.1, outside_value=-7.5)
    assert got[0, 0].item() == pytest.approx(-7.5)
    assert got[0, 1].item() == pytest.approx(-7.5)
    assert got[0, 2].item() == pytest.approx(1.0)


def test_sample_keeps_envs_separate():
    grid = torch.stack([torch.full((6, 6), 1.0), torch.full((6, 6), 5.0)])
    origins = torch.tensor([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    q = torch.tensor([[[0.0, 0.0]], [[20.0, 0.0]]])
    got = sample_height_grid(grid, q, origins, (-0.3, -0.3), 0.1, outside_value=-9.0)
    assert got[0, 0].item() == pytest.approx(1.0)
    assert got[1, 0].item() == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# the heightfield backend
# ---------------------------------------------------------------------------


def test_heightfield_scan_is_chassis_relative():
    """Flat ground under a chassis at height h must read h everywhere,
    whatever the absolute elevation of the terrain."""
    nx = ny = 40
    cell = 0.25
    lower = (-5.0, -5.0)
    pattern = nav_scan_pattern()
    origins = torch.zeros(1, 3)

    for terrain_z in (0.0, 2.5, -1.75):
        grid = torch.full((1, ny, nx), terrain_z)
        ref = torch.tensor([terrain_z + 0.30])          # chassis one wheel radius up
        pos = torch.tensor([[0.0, 0.0, terrain_z + 0.30]])
        scan = scan_from_heightfield(
            grid, pos, IDENT, pattern, origins, lower, cell, ref, clip=1.0,
        )
        assert scan.shape == (1, NAV_SCAN_CELLS)
        assert torch.allclose(scan, torch.full_like(scan, 0.30), atol=1e-5), terrain_z


def test_heightfield_scan_sees_a_pit_as_a_larger_distance():
    nx = ny = 40
    cell = 0.25
    lower = (-5.0, -5.0)
    grid = torch.zeros(1, ny, nx)
    grid[0, 18:22, 18:22] = -0.20                        # a pit right of centre
    pattern = nav_scan_pattern()
    ref = torch.tensor([0.30])
    scan = scan_from_heightfield(
        grid, torch.tensor([[0.0, 0.0, 0.30]]), IDENT, pattern,
        torch.zeros(1, 3), lower, cell, ref, clip=1.0,
    )
    assert scan.max().item() > 0.30 + 1e-3, "pit must read further away than flat ground"
    assert scan.min().item() == pytest.approx(0.30, abs=1e-3)


def test_heightfield_scan_clips():
    grid = torch.full((1, 40, 40), -50.0)
    scan = scan_from_heightfield(
        grid, torch.tensor([[0.0, 0.0, 0.30]]), IDENT, nav_scan_pattern(),
        torch.zeros(1, 3), (-5.0, -5.0), 0.25, torch.tensor([0.30]), clip=1.0,
    )
    assert scan.max().item() <= 1.0 and scan.min().item() >= -1.0


# ---------------------------------------------------------------------------
# the two backends must agree
# ---------------------------------------------------------------------------


def test_particle_and_heightfield_backends_agree_on_the_same_surface():
    """THE property. A dense particle sheet at a known height, and a
    heightfield of that same height, must produce the same scan. If these
    diverge, a skill trained on procedural terrain misreads MPM soil."""
    cell = 0.05
    nx = ny = 80
    lower = (-2.0, -2.0)
    surface = 0.0
    origins = torch.zeros(1, 3)
    pattern = dig_scan_pattern()
    ref = torch.tensor([0.50])
    pos = torch.tensor([[0.0, 0.0, 0.50]])

    # particles packed at least one per cell so every cell gets a sample
    step = cell / 2
    xs = torch.arange(lower[0], lower[0] + nx * cell, step)
    ys = torch.arange(lower[1], lower[1] + ny * cell, step)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    parts = torch.stack([gx.flatten(), gy.flatten(),
                         torch.full((gx.numel(),), surface)], dim=-1)
    penv = torch.zeros(parts.shape[0], dtype=torch.long)

    from_particles = scan_from_particles(
        parts, penv, origins, pos, IDENT, pattern,
        lower, cell, nx, ny, floor_height=-1.0, reference_z=ref, clip=2.0,
    )
    from_field = scan_from_heightfield(
        torch.full((1, ny, nx), surface), pos, IDENT, pattern,
        origins, lower, cell, ref, clip=2.0,
    )
    assert from_particles.shape == from_field.shape == (1, DIG_SCAN_CELLS)
    assert torch.allclose(from_particles, from_field, atol=1e-4), (
        from_particles.min(), from_particles.max(), from_field.min(), from_field.max()
    )


def test_particle_backend_tracks_an_excavated_hollow():
    cell = 0.05
    nx = ny = 80
    lower = (-2.0, -2.0)
    step = cell / 2
    xs = torch.arange(lower[0], lower[0] + nx * cell, step)
    ys = torch.arange(lower[1], lower[1] + ny * cell, step)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    x, y = gx.flatten(), gy.flatten()
    z = torch.zeros_like(x)
    z[(x.abs() < 0.3) & (y.abs() < 0.3)] = -0.18        # a scooped hollow
    parts = torch.stack([x, y, z], dim=-1)
    penv = torch.zeros(parts.shape[0], dtype=torch.long)

    scan = scan_from_particles(
        parts, penv, torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 0.5]]), IDENT,
        dig_scan_pattern(), lower, cell, nx, ny, -1.0, torch.tensor([0.5]), clip=2.0,
    )
    assert scan.max().item() == pytest.approx(0.5 + 0.18, abs=0.02)
    assert scan.min().item() == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# procedural terrain
# ---------------------------------------------------------------------------


def test_generated_terrain_shape_and_variety():
    g = torch.Generator().manual_seed(0)
    field = generate_excavation_terrain(8, 40, 30, 0.25, generator=g)
    assert field.shape == (8, 30, 40)
    assert torch.isfinite(field).all()
    # envs must differ, or the randomisation is not doing anything
    assert not torch.allclose(field[0], field[1])


def test_generated_terrain_relief_stays_near_what_this_machine_can_dig():
    """Terrain the excavator could not itself produce teaches the navigator to
    avoid obstacles it will never meet."""
    from luna_hifi_tasks.excavator import excavator as E

    g = torch.Generator().manual_seed(1)
    field = generate_excavation_terrain(64, 40, 30, 0.25, generator=g)
    per_env_relief = field.amax(dim=(1, 2)) - field.amin(dim=(1, 2))
    assert per_env_relief.median().item() < 8.0 * E.reach()["dig_depth"]
    assert per_env_relief.min().item() > 0.02, "some relief in every env"
