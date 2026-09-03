# soil.py
#
# Everything soil-specific that isn't scene config: the material mapping ported
# from newton_soil.py, and the two Warp kernels that turn MPM particles into
# policy inputs (height-map, bucket fill).
#
# STUB STATUS
#   SoilMaterial / to_material_cfg   real, same semantics as newton_soil.py
#   height_map_kernel                plausible first pass, unvalidated
#   bucket_fill_kernel               plausible first pass, unvalidated
#
# Both kernels take particle positions as a (num_envs, n_per_env, 3) array.
# That layout is what MPMSolverCfg(separate_worlds=True) produces: particles
# stored contiguously per world. If you run with a shared grid the layout is
# flat and you'll need an env-id array instead.

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import warp as wp

# VERIFY: import path for the MPM material config in your isaaclab_newton
# checkout. Field names (density, young_modulus, poisson_ratio, friction,
# yield_pressure, tensile_yield_ratio, yield_stress, hardening, dilatancy,
# viscosity, damping) match what NewtonMPMManager writes as mpm:* attributes.
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg


# =============================================================================
# MATERIALS
# =============================================================================


@dataclass
class SoilMaterial:
    """Geotechnical -> Newton MPM parameter mapping. See newton_soil.py.

    friction = tan(phi), yield_stress = cohesion in Pa. Neither is the old
    Drucker-Prager alpha or log-strain cohesion -- recalibrate, don't port.
    """

    density: float = 1600.0
    young_modulus: float = 1.0e7
    poisson_ratio: float = 0.3
    friction: float = 0.68
    yield_stress: float = 0.0
    yield_pressure: float = 1.0e12
    tensile_yield_ratio: float = 0.0
    hardening: float = 0.0
    dilatancy: float = 0.0
    viscosity: float = 0.0
    damping: float = 0.0

    @classmethod
    def from_soil(
        cls,
        density: float,
        young_modulus: float,
        poisson_ratio: float,
        friction_angle_deg: float,
        cohesion_pa: float = 0.0,
        **kwargs,
    ) -> "SoilMaterial":
        return cls(
            density=density,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            friction=float(np.tan(np.deg2rad(friction_angle_deg))),
            yield_stress=cohesion_pa,
            **kwargs,
        )

    def to_material_cfg(self) -> MPMParticleMaterialCfg:
        """The Isaac Lab config object. This replaces emit_soil()'s
        custom_attributes dict -- the MPM manager writes the mpm:* attributes
        itself from this."""
        return MPMParticleMaterialCfg(**asdict(self))


# Starting points for lunar-simulant work. Calibrate against your repose-angle
# and plate-sinkage data before believing any reward curve.
DENSE_REGOLITH = SoilMaterial.from_soil(1800.0, 2.0e7, 0.3, 42.0, cohesion_pa=800.0)
LOOSE_SAND = SoilMaterial.from_soil(1400.0, 5.0e6, 0.3, 32.0)


# =============================================================================
# HEIGHT-MAP  (STUB -- unvalidated)
# =============================================================================
#
# Rasterises particles into a per-env 2D grid of max-z. The grid is fixed in
# the env-local frame, i.e. it does NOT follow the rover. That's the simplest
# thing that gives the policy any terrain information at all. The upgrade is a
# rover-relative, yaw-aligned window around the bucket -- transform particle
# positions into the base frame before binning.


@wp.kernel
def height_map_kernel(
    particle_pos: wp.array2d(dtype=wp.vec3),   # (num_envs, n_per_env), env-local
    grid_origin: wp.vec2,                       # xy of cell (0,0), env-local
    cell_size: float,
    rows: int,
    cols: int,
    floor_z: float,
    height_map: wp.array2d(dtype=float),        # (num_envs, rows*cols), pre-filled with floor_z
):
    env, p = wp.tid()
    pos = particle_pos[env, p]

    i = int((pos[1] - grid_origin[1]) / cell_size)
    j = int((pos[0] - grid_origin[0]) / cell_size)
    if i < 0 or i >= rows or j < 0 or j >= cols:
        return

    wp.atomic_max(height_map, env, i * cols + j, pos[2])


# =============================================================================
# BUCKET FILL  (STUB -- unvalidated)
# =============================================================================
#
# Counts particle mass inside an axis-aligned box in the bucket's local frame.
# "Inside the bucket" is a proxy for "scooped" -- a real signal needs the
# bucket's cavity geometry, but a box the size of the cavity is a fair first
# approximation and costs one kernel.


@wp.kernel
def bucket_fill_kernel(
    particle_pos: wp.array2d(dtype=wp.vec3),   # (num_envs, n_per_env), env-local
    particle_mass: wp.array2d(dtype=float),    # (num_envs, n_per_env)
    bucket_xform: wp.array(dtype=wp.transform),  # (num_envs,), env-local
    cavity_lo: wp.vec3,                         # bucket-local box
    cavity_hi: wp.vec3,
    fill_mass: wp.array(dtype=float),           # (num_envs,), pre-zeroed
):
    env, p = wp.tid()
    local = wp.transform_point(wp.transform_inverse(bucket_xform[env]), particle_pos[env, p])

    if (
        local[0] < cavity_lo[0] or local[0] > cavity_hi[0]
        or local[1] < cavity_lo[1] or local[1] > cavity_hi[1]
        or local[2] < cavity_lo[2] or local[2] > cavity_hi[2]
    ):
        return

    wp.atomic_add(fill_mass, env, particle_mass[env, p])
