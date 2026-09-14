"""Tests for the excavator MDP sensors.

These cover the two quantities every policy depends on. Both are checked
against brute-force references rather than against themselves: a vectorised
scatter that agrees with a Python loop over every particle is a scatter whose
index arithmetic is right.

Run: python -m pytest tests/test_sensors.py -q
"""

from __future__ import annotations

import math

import pytest
import torch

from luna_hifi_tasks.excavator.mdp.sensors import (
    drum_fill_fraction,
    drum_fill_mass,
    heightmap_to_obs,
    quat_apply_inverse,
    soil_heightmap,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# quaternion helper
# ---------------------------------------------------------------------------


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(x,y,z,w) -> 3x3 rotation matrix, built from the definition.

    Isaac Lab 3.x order. See the note on quat_apply_inverse."""
    x, y, z, w = q
    return torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=q.dtype)


def _quat_about_y(angle: float) -> torch.Tensor:
    return torch.tensor([0.0, math.sin(angle/2), 0.0, math.cos(angle/2)])


def test_quat_apply_inverse_matches_transpose_of_rotation_matrix():
    for angle in (0.0, 0.3, -1.2, 2.7, math.pi):
        for axis in range(3):
            half = angle / 2
            q = torch.zeros(4)
            q[3] = math.cos(half)             # w is LAST in Isaac Lab 3.x
            q[axis] = math.sin(half)
            R = _quat_to_mat(q)
            v = torch.randn(5, 3)
            got = quat_apply_inverse(q.unsqueeze(0).expand(5, 4), v)
            want = v @ R                      # v @ R == (R^T @ v^T)^T
            assert torch.allclose(got, want, atol=1e-5), (angle, axis)


def test_identity_quaternion_is_xyzw_not_wxyz():
    """Pins the convention. Under wxyz, [0,0,0,1] is a 180 deg turn about z and
    this test fails loudly instead of the reward quietly training on a rotated
    drum frame."""
    ident = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    v = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(quat_apply_inverse(ident, v), v, atol=1e-6)


# ---------------------------------------------------------------------------
# height map
# ---------------------------------------------------------------------------


def _heightmap_bruteforce(pos, env, origins, lower, cell, nx, ny, floor):
    E = origins.shape[0]
    out = torch.full((E, ny, nx), floor, dtype=pos.dtype)
    for i in range(pos.shape[0]):
        e = int(env[i])
        lx = float(pos[i, 0] - origins[e, 0])
        ly = float(pos[i, 1] - origins[e, 1])
        lz = float(pos[i, 2] - origins[e, 2])
        gx = math.floor((lx - lower[0]) / cell)
        gy = math.floor((ly - lower[1]) / cell)
        if 0 <= gx < nx and 0 <= gy < ny:
            out[e, gy, gx] = max(float(out[e, gy, gx]), lz)
    return out


def test_heightmap_matches_bruteforce_random():
    E, P, nx, ny, cell = 4, 3000, 11, 7, 0.08
    lower = (-0.4, -0.3)
    origins = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0],
                            [0.0, 5.0, 0.0], [5.0, 5.0, 0.0]])
    env = torch.randint(0, E, (P,))
    # spread wider than the grid so the out-of-bounds path is exercised
    local = torch.rand(P, 3) * torch.tensor([1.6, 1.2, 0.4]) - torch.tensor([0.8, 0.6, 0.1])
    pos = local + origins[env]

    got = soil_heightmap(pos, env, origins, lower, cell, nx, ny, floor_height=-0.05)
    want = _heightmap_bruteforce(pos, env, origins, lower, cell, nx, ny, -0.05)
    assert got.shape == (E, ny, nx)
    assert torch.allclose(got, want, atol=1e-6)


def test_heightmap_empty_cells_read_floor_not_zero():
    """An excavated cell and a never-touched cell must not look alike."""
    origins = torch.zeros(1, 3)
    pos = torch.tensor([[0.02, 0.02, 0.30]])
    env = torch.zeros(1, dtype=torch.long)
    hm = soil_heightmap(pos, env, origins, (0.0, 0.0), 0.1, 3, 3, floor_height=-0.25)
    assert hm[0, 0, 0] == pytest.approx(0.30)
    assert (hm[0, 1:, :] == pytest.approx(-0.25))
    assert hm.min() == pytest.approx(-0.25)


def test_heightmap_takes_max_not_last():
    origins = torch.zeros(1, 3)
    pos = torch.tensor([[0.05, 0.05, 0.1], [0.05, 0.05, 0.9], [0.05, 0.05, 0.4]])
    env = torch.zeros(3, dtype=torch.long)
    hm = soil_heightmap(pos, env, origins, (0.0, 0.0), 0.1, 2, 2, floor_height=0.0)
    assert hm[0, 0, 0] == pytest.approx(0.9)


def test_heightmap_out_of_range_particles_are_dropped_not_clamped():
    """Clamping would build a fake wall of soil on the boundary cells."""
    origins = torch.zeros(1, 3)
    inside = torch.tensor([[0.05, 0.05, 0.1]])
    outside = torch.tensor([[9.0, 9.0, 5.0], [-9.0, -9.0, 5.0]])
    pos = torch.cat([inside, outside])
    env = torch.zeros(3, dtype=torch.long)
    hm = soil_heightmap(pos, env, origins, (0.0, 0.0), 0.1, 2, 2, floor_height=0.0)
    assert hm.max() == pytest.approx(0.1)       # the 5.0 never lands anywhere


def test_heightmap_envs_do_not_leak_into_each_other():
    origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    pos = torch.tensor([[0.05, 0.05, 0.7], [10.05, 0.05, 0.2]])
    env = torch.tensor([0, 1])
    hm = soil_heightmap(pos, env, origins, (0.0, 0.0), 0.1, 2, 2, floor_height=0.0)
    assert hm[0, 0, 0] == pytest.approx(0.7)
    assert hm[1, 0, 0] == pytest.approx(0.2)


def test_heightmap_to_obs_is_relative_and_clipped():
    hm = torch.tensor([[[0.0, 1.0], [-1.0, 0.05]]])
    obs = heightmap_to_obs(hm, reference=0.0, clip=0.5, scale=2.0)
    assert obs.shape == (1, 4)
    assert obs.max() <= 1.0 + 1e-6 and obs.min() >= -1.0 - 1e-6
    # same terrain at a different elevation must give the same observation
    a = heightmap_to_obs(hm, reference=0.0)
    b = heightmap_to_obs(hm + 3.0, reference=3.0)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# drum fill
# ---------------------------------------------------------------------------


def _fill_bruteforce(pos, env, mass, dpos, dquat, r, hl, E):
    out = torch.zeros(E, dtype=pos.dtype)
    for i in range(pos.shape[0]):
        e = int(env[i])
        R = _quat_to_mat(dquat[e])
        local = R.T @ (pos[i] - dpos[e])
        if local[0]**2 + local[2]**2 < r*r and abs(local[1]) < hl:
            out[e] += mass if isinstance(mass, float) else float(mass[i])
    return out


def test_drum_fill_matches_bruteforce_under_rotation():
    E, P = 3, 2000
    r, hl, mass = 0.17, 0.475, 0.05
    dpos = torch.tensor([[0.0, 0.0, 0.5], [4.0, 0.0, 0.5], [8.0, 0.0, 0.5]])
    # a pitched arm and a spun drum -- both are rotations about y
    dquat = torch.stack([_quat_about_y(0.0), _quat_about_y(0.8), _quat_about_y(-1.9)])
    env = torch.randint(0, E, (P,))
    pos = dpos[env] + (torch.rand(P, 3) - 0.5) * torch.tensor([1.2, 1.4, 1.2])

    got = drum_fill_mass(pos, env, mass, dpos, dquat, r, hl, E)
    want = _fill_bruteforce(pos, env, mass, dpos, dquat, r, hl, E)
    assert torch.allclose(got, want, atol=1e-5), (got, want)
    assert got.sum() > 0, "test is vacuous if nothing landed inside"


def test_drum_fill_axis_is_local_y():
    """The cavity is a cylinder about the drum's own y. A particle far out
    along y is outside; the same distance along x is inside."""
    dpos = torch.zeros(1, 3)
    dquat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    r, hl = 0.17, 0.475
    along_y = torch.tensor([[0.0, 0.60, 0.0]])     # beyond half-length
    along_x = torch.tensor([[0.10, 0.0, 0.0]])     # inside the bore
    env = torch.zeros(1, dtype=torch.long)
    assert drum_fill_mass(along_y, env, 1.0, dpos, dquat, r, hl, 1).item() == 0.0
    assert drum_fill_mass(along_x, env, 1.0, dpos, dquat, r, hl, 1).item() == 1.0


def test_drum_fill_follows_a_pitched_arm():
    """A particle sitting in the cavity must stay counted when the arm pitches.
    A world-axis-aligned box test would lose it; this is what that guards."""
    r, hl = 0.17, 0.475
    env = torch.zeros(1, dtype=torch.long)
    for angle in (0.0, 0.4, 0.8, -0.55):
        q = _quat_about_y(angle)
        R = _quat_to_mat(q)
        centre = torch.tensor([[1.2, 0.0, 0.45]])
        # 8 cm off-axis in the DRUM's frame, pushed out to world
        p = centre + (R @ torch.tensor([0.08, 0.0, 0.0])).unsqueeze(0)
        got = drum_fill_mass(p, env, 1.0, centre, q.unsqueeze(0), r, hl, 1)
        assert got.item() == pytest.approx(1.0), angle


def test_drum_fill_per_particle_mass_tensor():
    dpos = torch.zeros(2, 3)
    dpos[1, 0] = 5.0
    dquat = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [5.0, 0.0, 0.0]])
    env = torch.tensor([0, 0, 1])
    mass = torch.tensor([2.0, 3.0, 7.0])
    got = drum_fill_mass(pos, env, mass, dpos, dquat, 0.17, 0.475, 2)
    assert got[0].item() == pytest.approx(5.0)
    assert got[1].item() == pytest.approx(7.0)


def test_fill_fraction_is_not_clamped_above_one():
    """A reading over 1.0 means the capacity estimate is wrong or particles are
    inside the shell. Clamping would hide both."""
    frac = drum_fill_fraction(torch.tensor([200.0]), capacity_kg=155.0)
    assert frac.item() > 1.0


# ---------------------------------------------------------------------------
# integration with the real asset geometry
# ---------------------------------------------------------------------------


def _particle_bed(spacing=0.025, half_x=0.6, half_y=0.7, depth=0.5):
    """A VOLUME of particles, the way an MPM bed actually is. A flat sheet
    would not do: lowered past its own centre, the chord a sheet cuts through
    the cavity starts shrinking again, so fill would peak and fall."""
    xs = torch.arange(-half_x, half_x, spacing)
    ys = torch.arange(-half_y, half_y, spacing)
    zs = torch.arange(-depth, 0.0, spacing)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)


def test_fill_rises_monotonically_as_the_drum_is_lowered_into_a_bed():
    """End to end against the real drum dimensions, over the real dig range."""
    from luna_hifi_tasks.excavator import excavator as E

    bore = E.DRUM_RADIUS - E.DRUM_WALL_T
    bed = _particle_bed()
    env = torch.zeros(bed.shape[0], dtype=torch.long)
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    # Drum-CENTRE heights, strictly descending. Note the centre at full dig is
    # +0.012, not negative: reach()["dig_depth"] measures the drum's lowest
    # point below grade, which is one DRUM_RADIUS below the centre.
    depths = [0.30, 0.20, 0.10, 0.0, -0.05]
    assert all(b < a for a, b in zip(depths, depths[1:])), "sweep must descend"
    fills = []
    for z in depths:
        centre = torch.tensor([[0.0, 0.0, float(z)]])
        fills.append(drum_fill_mass(bed, env, 1.0, centre, q, bore, E.DRUM_HALF_LEN, 1).item())

    assert fills[0] == 0.0, f"drum clear of the bed should read empty, got {fills[0]}"
    assert all(b >= a for a, b in zip(fills, fills[1:])), fills
    assert fills[-1] > 0, fills


def test_fill_never_exceeds_the_analytic_bore_capacity():
    """Guards the capacity constant the reward normalises by. If a fully
    buried drum reads more than pi r^2 L worth of particles, the bore radius
    or half-length in the env cfg disagrees with the asset."""
    from luna_hifi_tasks.excavator import excavator as E

    spacing = 0.025
    bore = E.DRUM_RADIUS - E.DRUM_WALL_T
    bed = _particle_bed(spacing=spacing, depth=0.9)
    env = torch.zeros(bed.shape[0], dtype=torch.long)
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    # bury the drum completely
    centre = torch.tensor([[0.0, 0.0, -0.45]])
    particle_volume = spacing ** 3
    got_volume = drum_fill_mass(
        bed, env, particle_volume, centre, q, bore, E.DRUM_HALF_LEN, 1
    ).item()
    analytic = math.pi * bore ** 2 * (2 * E.DRUM_HALF_LEN)

    # A cubic lattice cannot exactly fill a cylinder, so the count sits a few
    # percent UNDER the analytic volume. Reading over it is the failure that
    # matters: it would mean the bore radius or half-length is too generous and
    # the reward is counting soil that is really inside the shell wall.
    assert got_volume <= analytic * 1.001, (got_volume, analytic)
    assert got_volume == pytest.approx(analytic, rel=0.05), (got_volume, analytic)
    assert got_volume == pytest.approx(E.reach()["drum_bore_volume"], rel=0.05)


# ---------------------------------------------------------------------------
# Newton adapters
# ---------------------------------------------------------------------------


class _FakeMaterial:
    density = 1800.0


class _FakeGridCfg:
    lower = (-0.4, -0.3, 0.025)
    upper = (1.1, 0.3, 0.085)
    voxel_size = 0.05
    particles_per_cell = 1.0
    material = _FakeMaterial()
    mass = None


def test_particle_mass_matches_the_spawner_arithmetic():
    """Reproduces isaaclab_newton's derivation: resolution is ceil()ed per
    axis, so cell volume is NOT voxel_size**3 / particles_per_cell."""
    from luna_hifi_tasks.excavator.mdp.sensors import mpm_grid_particle_mass

    cfg = _FakeGridCfg()
    extent = [u - l for u, l in zip(cfg.upper, cfg.lower)]
    res = [max(math.ceil(cfg.particles_per_cell * e / cfg.voxel_size), 1) for e in extent]
    cell_vol = 1.0
    for e, r in zip(extent, res):
        cell_vol *= e / r
    assert mpm_grid_particle_mass(cfg) == pytest.approx(cell_vol * 1800.0)

    naive = cfg.voxel_size ** 3 * 1800.0
    assert mpm_grid_particle_mass(cfg) < naive, "ceil() must shrink the real particle"


def test_particle_mass_honours_an_explicit_override():
    from luna_hifi_tasks.excavator.mdp.sensors import mpm_grid_particle_mass

    cfg = _FakeGridCfg()
    cfg.mass = 0.123
    assert mpm_grid_particle_mass(cfg) == pytest.approx(0.123)


class _FakeProxy:
    def __init__(self, t): self.torch = t


class _FakeData:
    def __init__(self, t): self.particle_pos_w = _FakeProxy(t)


class _FakeMPM:
    def __init__(self, E, P):
        self.num_instances = E
        self.particles_per_object = P
        self.data = _FakeData(torch.arange(E * P * 3, dtype=torch.float32).reshape(E, P, 3))


def test_mpm_particle_state_builds_the_env_index_by_stride():
    """Particles are already (E, P, 3). The env index is repeat_interleave over
    a fixed stride, NOT a divide over a flat array."""
    from luna_hifi_tasks.excavator.mdp.sensors import mpm_particle_state

    pos, env = mpm_particle_state(_FakeMPM(3, 5))
    assert pos.shape == (15, 3)
    assert env.tolist() == [0]*5 + [1]*5 + [2]*5
    assert torch.equal(pos[5], torch.tensor([15.0, 16.0, 17.0]))


def test_mpm_particle_state_rejects_an_unexpected_layout():
    from luna_hifi_tasks.excavator.mdp.sensors import mpm_particle_state

    bad = _FakeMPM(3, 5)
    bad.particles_per_object = 7          # disagrees with the array
    with pytest.raises(RuntimeError, match="unexpected MPM particle layout"):
        mpm_particle_state(bad)
