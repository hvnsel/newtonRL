# excavate_env_cfg.py
#
# Excavation with an implicit-MPM regolith bed
# Gravity is lunar gravity. See excavator_cfg.py.

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
)
from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from ..excavator import (
    ARM_LEN,
    ARM_RANGE,
    CHASSIS_Z,
    MAST_TOP_Z,
    ROTOR_HALF_LEN,
    ROTOR_TIP_R,
    SHROUD_END_T,
    SHROUD_OUT_R,
    SHROUD_RANGE,
    WHEEL_HALF_W,
    WHEEL_Y,
)
from ..excavator_cfg import (
    DRUM_CONTACT_BODIES_REGEX,
    EXCAVATOR_CFG,
    EXCAVATOR_PRIM_REGEX,
    LUNAR_GRAVITY,
    SOIL_CONTACT_BODIES_REGEX,
    SPAWN_Z,
    WHEEL_CONTACT_BODIES_REGEX,
)
from ..mdp.observations import (
    NAV_SCAN_CELLS,
    critic_state_spec,
    excavate_obs_spec,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RIGID_ENTRY = "rover" # Entry name for the rigid body in the coupler
MPM_ENTRY = "soil" # Entry name for the MPM object in the coupler

# Heightmap cell: what the bed grid, the terrain scans and the commanded cut
# depth are quantised to. Also the particle spacing, via MPM_VOXEL_SIZE and
# particles_per_cell below.
VOXEL_SIZE = 0.04
MPM_COLLIDER_MARGIN = 0.5 * VOXEL_SIZE

# The MPM solver's background grid, which is a different thing and wants to be
# COARSER than the particle spacing. MPMSolverCfg.voxel_size is documented as
# "Size of the MPM grid voxels [m]" and nothing else -- it does not set
# particle spacing, which comes from MPMGridCfg's lattice, and it does not set
# the collider margin, which is MPM_COLLIDER_MARGIN above. Newton's own
# default is 0.1 m.
#
# At 0.04 it equalled the particle spacing, which is what starved every cell
# to a single particle and let the bed collapse under its own weight. 0.08
# with particles_per_cell 2.0 puts 8 particles in each solver cell while the
# spawn lattice stays at 4 cm: the same 50,750 particles as before, the same
# 0.0857 kg each, so nothing downstream that counts kilograms changes. The
# grid itself drops to an eighth of the cells.
MPM_VOXEL_SIZE = 0.08
# Particles per cell PER AXIS: the spawn lattice is ceil(ppc * extent / voxel)
# on each axis, so 2 is 8 particles per cell in 3D, which is what MPM wants.
#
# It was 1. One particle per cell leaves the deformation gradient
# under-determined -- the background grid carries modes no particle resists --
# so the material loses volume under any load including its own weight. A
# replay measured the bed falling 25.05 -> 13.33 cm, a factor of 1.88, over
# soil the machine never approached, while the greatest vertical stress in a
# 25 cm bed at lunar gravity is 2430 * 0.25 = 608 Pa against 500 Pa of
# cohesion at 40 degrees of friction, which cannot fail. The machine was not
# sinking into the soil; the soil was collapsing underneath it. Bearing
# capacity says 492 kg at 1.62 m/s2 is 6.5 kPa under a wheel against ~60 kPa
# of capacity, a factor of nine, so it should settle about a centimetre.
#
# cell_center placement with jitter 0.0 compounds it: every particle crosses a
# cell boundary on the same step, so the cell-crossing error is coherent
# rather than averaging out.
MPM_PARTICLES_PER_CELL = 2.0
MPM_VISUAL_COLOR = (0.62, 0.55, 0.45)

# The bed extent lives on the env cfg as bed_x / bed_y / bed_depth
BED_FLOOR_Z = 0.0

# friction is ~tan(phi), yield_stress is cohesion in Pa, yield_pressure caps compression.
SOIL_MATERIAL = MPMParticleMaterialCfg(
    density=1800.0,
    friction=0.84,
    yield_pressure=1.0e12,
)

BORE_RADIUS = ROTOR_TIP_R
BORE_HALF_LEN = ROTOR_HALF_LEN

# The ground the drum is working: scan cells inside this rectangle, in the
# drum's own frame, are the ones a cut brings down to the target height.
FOOTPRINT_HALF_X = SHROUD_OUT_R
FOOTPRINT_HALF_Y = ROTOR_HALF_LEN
BORE_VOLUME = math.pi * BORE_RADIUS ** 2 * (2.0 * BORE_HALF_LEN)

# The mass the rotor's swept cylinder holds packed solid.
# Fill is normalized and scored against target_load_kg.
DRUM_CAPACITY_KG = BORE_VOLUME * SOIL_MATERIAL.density

DIG_OBS = excavate_obs_spec()
DIG_CRITIC = critic_state_spec(terrain_cells=NAV_SCAN_CELLS)


def particles_per_env(lower, upper, voxel: float, per_cell: float) -> int:
    """Calculate the number of partices in an env"""
    n = 1
    for lo, hi in zip(lower, upper):
        n *= max(int(math.ceil(per_cell * (hi - lo) / voxel)), 1)
    return n


def _next_pow2(n: int) -> int:
    return 1 << max(int(n - 1).bit_length(), 1)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

SOIL_CFG = MPMObjectCfg(
    prim_path="{ENV_REGEX_NS}/Soil",
    init_state=MPMObjectCfg.InitialStateCfg(),
    spawn=MPMGridCfg(
        # Placeholder bounds for the MPM grid. these will be overwritten in __post_init__
        lower=(-1.0, -1.0, MPM_COLLIDER_MARGIN), 
        upper=(1.0, 1.0, MPM_COLLIDER_MARGIN + 0.25),
        voxel_size=MPM_VOXEL_SIZE,
        particles_per_cell=MPM_PARTICLES_PER_CELL,
        particle_placement="cell_center",
        jitter=0.0,
        material=SOIL_MATERIAL,
        visual_color=MPM_VISUAL_COLOR,
    ),
)

def _mpm_ground() -> RigidObjectCfg:
    """Hidden kinematic slab giving the MPM entry a floor.

    The MPM entry sees only the bodies listed on its CouplerEntryCfg, so this
    slab is the floor under every particle.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MPMGround",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 3.0, 0.10),
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=MPM_COLLIDER_MARGIN,
                contact_gap=0.0,
            ),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.8, dynamic_friction=0.7),
            visible=False,
        ),
    )


def _bed_wall(name: str) -> RigidObjectCfg:
    """Hidden kinematic plate retaining one side of the bed.

    Without these the bed is a box with four free faces, and a cohesionless
    soil cannot hold one: the angle of repose equals the friction angle, so a
    0.25 m face slumps 0.30 m and the usable width shrinks by that much from
    each side. Real regolith continues past where a test bed stops, and a
    retaining plate is a closer approximation of that than a collapsing face.

    They also pay for themselves in memory. The grid is sparse with zero
    padding, so it follows the particles: soil that spreads outward grows the
    grid and spends active cells, which is the budget that caps env count.

    Sized and placed in __post_init__, and listed by name on the MPM coupler
    entry the same way MPMGround is. The top sits flush with the soil surface
    so the plates retain the bed without standing in the machine's way --
    leaving the bed is caught geometrically in _failed instead.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BedWall" + name,
        init_state=RigidObjectCfg.InitialStateCfg(),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=UsdPhysicsRigidBodyCfg(
                rigid_body_enabled=True, kinematic_enabled=True
            ),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=MPM_COLLIDER_MARGIN,
                contact_gap=0.0,
            ),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=0.5, dynamic_friction=0.5
            ),
            visible=False,
        ),
    )


def _wheel_rail(name: str) -> RigidObjectCfg:
    """Hidden kinematic rail carrying one wheel track at grade.

    The soil develops no bearing capacity under the wheels -- see
    DRUM_CONTACT_BODIES_REGEX for what was ruled out -- so the wheels are
    taken out of the soil coupling and given something rigid to ride on. A
    rail runs the length of the bed under one track with its top flush with
    the soil surface, and it is named on the rover's coupler entry and on no
    other: the MPM entry sees only the bodies listed on it, so the soil passes
    straight through the rail and keeps its full extent while the wheels ride
    on it.

    The consequence to be honest about: wheel-terrain interaction becomes
    rigid contact with no sinkage and no motion resistance. The soil the
    drums cut is unaffected, which is what the dig policy is learning.

    Sized and placed in __post_init__. Set wheel_rails False to park them
    below the floor and put the wheels back on soil.
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WheelRail" + name,
        init_state=RigidObjectCfg.InitialStateCfg(),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=UsdPhysicsRigidBodyCfg(
                rigid_body_enabled=True, kinematic_enabled=True
            ),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=MPM_COLLIDER_MARGIN,
                contact_gap=0.0,
            ),
            # The traction the wheels would have had on regolith.
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=1.0, dynamic_friction=0.9
            ),
            visible=False,
        ),
    )


@configclass
class ExcavateSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    excavator: ArticulationCfg = EXCAVATOR_CFG
    mpm_ground: RigidObjectCfg = _mpm_ground()
    bed_wall_xlo: RigidObjectCfg = _bed_wall("Xlo")
    bed_wall_xhi: RigidObjectCfg = _bed_wall("Xhi")
    bed_wall_ylo: RigidObjectCfg = _bed_wall("Ylo")
    bed_wall_yhi: RigidObjectCfg = _bed_wall("Yhi")
    wheel_rail_ylo: RigidObjectCfg = _wheel_rail("Ylo")
    wheel_rail_yhi: RigidObjectCfg = _wheel_rail("Yhi")
    soil: MPMObjectCfg = SOIL_CFG

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@configclass
class ExcavatorExcavateEnvCfg(DirectRLEnvCfg):

    decimation = 4 # Render every 4 simulation steps
    # 21 s of commanded cut after settle_seconds of unpaid settling below.
    episode_length_s = 24.0
    # The bed is re-spawned on its spawn lattice every soil_reset_every
    # episodes and then consolidates under its own weight. Measured off a
    # replay: 7.23 cm of subsidence over soil the machine never went near,
    # against a commanded trim of one 4 cm voxel. Every progress term is a
    # difference against the surface at episode start, so a bed still settling
    # reads as soil excavated -- 111.2 kg of "removal" against 117.1 kg of
    # pure settlement, and 642 of 675 work-area cells "cleared" without the
    # machine reaching the work area at all. shape_done was firing on
    # subsidence, which is why it sat at 0.49 across three different rewards.
    #
    # So the surface is sampled after the bed has stopped moving, not when it
    # is spawned. _cut_fresh holds for this long, which already zeroes every
    # delta, blocks shape_done and keeps re-capturing the reference surface,
    # so the reference ends up being the settled bed. Set it from the
    # subsidence curve: long enough that the remaining drift over an episode
    # is small against a voxel.
    settle_seconds: float = 3.0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=decimation,
        gravity=LUNAR_GRAVITY,
    )

    scene: ExcavateSceneCfg = ExcavateSceneCfg(
        num_envs=16,
        # Beds are 8 m long, so origins have to stay at least that far apart.
        # Beyond that, keep them as close as possible: the MPM sparse grid's
        # node tree is budgeted by the span it has to cover, not by particle
        # count, and it is the first thing to run out. At 4 envs a 12.0 m
        # spacing spans 20 x 14 m and the rebuild fails every step with
        # LOWER and UPPER internal node capacity exceeded; 9.0 m spans
        # 17 x 11 m and is clean. Raising the node counts does not help --
        # only shrinking the span does, which is what the solver's own error
        # suggests. grid_cap_multiplier is unrelated: it sizes active voxels,
        # and VOXEL_CAPACITY_EXCEEDED is never among the flags.
        env_spacing=9.0,
        replicate_physics=True,
    )

    # --- bed, in the env frame ---
    #
    # The machine spawns at the origin facing +x. With spawn_on_bed the wheels
    # rest on the bed surface and both drums start over soil: the front one
    # meets fresh regolith as it drives, the rear trails through the trench.
    # 5.8 m, down from 8.0. The machine spans +-1.81 m at the dig angle and
    # stands on the bed, spawn jitter is +-0.55, and the work area now reaches
    # 0.9 m rather than 2.3, so the chassis origin stays inside
    # [-0.55, 1.45]. With bed_margin_x at 1.81 that needs -2.36 and 3.26.
    # Worth doing because throughput goes as 1/particles-per-env.
    bed_x: tuple[float, float] = (-2.4, 3.4)
    bed_y: tuple[float, float] = (-1.0, 1.0)
    bed_depth: float = 0.25
    # Retaining plates outside each bed face. A few voxels is enough; the
    # plates are kinematic and only ever take a normal load.
    bed_wall_thickness: float = 0.10
    # Only meaningful for a bed the machine stands on and cuts from within.
    # Where the bed is a pile reached from outside, a plate on the near face
    # sits exactly where the drum comes down, so the plates are parked below
    # the floor instead of being removed, which would change the asset count
    # the coupler is built against.
    bed_walls: bool = True

    # How far the chassis origin may sit from a bed face before the episode
    # is a failure. At the dig angle the machine reaches 1.81 m fore and aft
    # of its origin and 0.6 m to each side, so these are its own half-extents:
    # the episode ends as the machine runs out of bed rather than when it
    # touches a plate, which needs no contact query and cannot be missed.
    bed_margin_x: float = 1.81
    bed_margin_y: float = 0.60
    # Particle spacing, and half of it is the margin the coupler inflates
    # every collider by. The 0.185 m gap between vane tips is 5.2 grains clear
    # at 0.03, 2.7 at 0.05; granular material arches below about four. Particle
    # count goes as the inverse cube.
    voxel_size: float = VOXEL_SIZE
    # The solver grid, deliberately coarser than the particle spacing.
    mpm_voxel_size: float = MPM_VOXEL_SIZE
    particles_per_cell: float = MPM_PARTICLES_PER_CELL

    # Gauss-Seidel sweeps per MPM solve, and the tolerance that would stop it
    # early. Both were buried in the MPMSolverCfg block below; they are fields
    # so a run can sweep them, and the long notes at their use sites explain
    # the traps (a value under solve_granularity = 25 runs ZERO sweeps).
    #
    # Measured with --mpm_verbose: the solve runs to the cap on nearly every
    # step. res_l2 lands at 5e-7..6.5e-6, already 2-20x under tolerance, while
    # res_linf sits at 1e-5..1.2e-4 and is what fails -- a handful of
    # degenerate nodes, most likely the ~0.6 % of particles with no strain
    # node at all, which carry no physics and cannot converge. The only two
    # solves in a 1 s run that stopped early (161 and 206 sweeps) both did so
    # the moment res_linf dipped below 1e-5.
    #
    # So this cap IS the solve cost, not a safety net on it, and the honest
    # value is the smallest that does not move the answer.
    mpm_max_iterations: int = 250
    # Tolerance has to sit BELOW the physical per-step strain rho*g*H/K it is
    # meant to resolve. At the old 4x soil density that signal was 2.4e-5 and
    # 1e-5 cleared it; at lunar regolith's 1500 kg/m3 it is 6.1e-6 and 1e-5
    # does not, which is exactly the fault the note below records for 1e-4.
    # Costs nothing in time: the solve never reaches tolerance anyway.
    mpm_tolerance: float = 2.0e-6
    spawn_on_bed: bool = True

    # How far past the bed, on every side, the MPM floor slab and the height
    # grid reach. Soil spreads outside the box it was spawned in; a single 12 s
    # crawl pushes the near edge back 0.20 m. The slab is the floor under it
    # and the height grid is what rasterises it into the terrain scan.
    mpm_floor_margin: float = 1.0
    heightmap_margin: float = 1.0

    # Sparse-grid active cells per particle, as absolute totals over all envs.
    # Overrunning them is an illegal access: a CUDA 700 storm, or 0xC0000374 on
    # Windows. A cell is tens of bytes, so 2^20 of them is under 100 MB.
    # Active voxels allocated per particle. 8 was right when a cell held one
    # particle; at 8 particles per cell the same soil occupies an eighth as
    # many cells, so leaving this at 8 would allocate eight times the grid for
    # the same bed and walk straight back into the memory wall. 1.0 keeps the
    # allocation where it is now -- 2^19 active at one env -- for 7.4x the
    # particles.
    grid_cap_multiplier: float = 1.0

    # Which bodies are coupled to the soil. Each brings every one of its
    # geoms; the drums are 38 apiece.
    #
    # None follows wheel_rails, which is what you want: the cutting end alone
    # when the wheels have rails to ride, every soil-contacting body when they
    # do not. Set it to a regex only to override that pairing deliberately.
    soil_contact_regex: str | None = None

    # Rails under the wheel tracks, tops flush with the soil surface, carrying
    # the wheels at grade because the soil will not.
    #
    # False parks them below the floor and -- through soil_contact_regex above
    # -- puts the wheels back in the soil coupling, which is the configuration
    # that sinks. That is the one switch: the two settings are paired on
    # purpose, because rails off with the wheels still uncoupled would leave
    # them resting on nothing and drop the machine to the bed floor at once.
    # False: the wheels belong on soil. True parks them on rigid beams, which
    # was a workaround from when the coupling could not hold the machine up at
    # all, and it makes every soil result moot -- the startup line reads
    # "wheels on rigid rails" and "soil coupling: drums and shrouds", with the
    # wheels coupled to nothing. Training against that would train against the
    # workaround. Everything measured this session used --no_rails.
    wheel_rails: bool = False

    # How far outboard of the cutting end a rail's inner face starts.
    #
    # The rotor ends at ROTOR_HALF_LEN = 0.475 but the shroud end plates close
    # the pockets past it, so the real axial extent is 0.487 -- 12 mm inside
    # the tyre's inboard edge at 0.475. The cutting end and the wheels already
    # overlap in y; what keeps them apart is separation in x, which is the
    # machine's own design and is noted as such in excavator.py. A rail runs
    # the length of the bed and so has no x separation to rely on, and that
    # makes this clearance the only thing stopping a drum from grinding into
    # one. It must exceed two collider margins, one on each face: at
    # MPM_COLLIDER_MARGIN 0.02 that is 0.04, and scripts/rail_check.py
    # asserts it rather than trusting the arithmetic here.
    wheel_rail_clearance: float = 0.06

    # Rigid-solver buffers, per env, fixed size. Overrunning one is an illegal
    # memory access: a CUDA 700 storm, or 0xC0000374 on Windows.
    #
    # Contacts scale with collider count and with how much soil touches them.
    # The shrouded rotor carries 136 geoms, 106 of them coupled. A contact is a
    # couple of hundred bytes, so 32768 is about 6.6 MB.
    rigid_njmax: int = 8192
    rigid_nconmax: int = 32768

    # Capturing the step as a CUDA graph reserves a memory pool covering every
    # allocation the capture touched, and MPM's transients make that pool
    # gigabytes. Instantiating it is what fails first on a 24 GB card: the
    # capture itself succeeds, then wp_cuda_graph_create_exec reports CUDA
    # error 2. Turning it off costs per-step launch overhead, which is noise
    # against a step that already takes seconds at this particle count.
    use_cuda_graph: bool = True

    # Active cells per leaf node, as a right shift: 7 is one leaf per 128
    # cells, four times the minimum for 8^3 blocks. Each step down costs 512
    # cells of storage.
    #
    # 3 rather than 7, because the node tree is what runs out first and the
    # floors it otherwise sits at are sized for one undisturbed bed. Measured
    # at 4 envs over 18 s of real digging: shift 7 and 4 both fail the rebuild
    # with LEAF, LOWER and UPPER capacity exceeded, 3 is clean and still fits
    # 24 GB. The shift has to be judged against worked soil -- a short run
    # never disturbs enough of the bed to grow the tree, and will pass at any
    # setting. grid_cap_multiplier is a separate budget and cannot substitute:
    # it sizes active voxels, and VOXEL_CAPACITY_EXCEEDED is never set.
    grid_leaf_shift: int = 3

    # Floor on upper internal nodes, which the shift alone cannot reach: upper
    # derives from lower >> 3, so shift 3 leaves it at 1024 and raising it
    # further would mean shift 2, quadrupling the leaf array to move a node
    # count that is small and cheap. Shift 3 took the rebuild status from 7
    # (leaf, lower and upper) to 4 (upper alone), so this is the remainder.
    grid_upper_min: int = 4096

    # Whether the MPM solver partitions its grid per world. One world runs
    # clean; every multi-world configuration fails the rebuild on lower and
    # upper internal nodes while those are provisioned about a thousand times
    # over what the span geometrically needs, which points at the multi-world
    # path rather than at any budget. Exposed so the two can be compared.
    # "none" rather than "auto". A grid-backed warm start is what makes a
    # masked reset illegal on a shared grid: the check escapes only on a None
    # mask, which newton_manager never passes, so separate_worlds off is
    # otherwise unusable under Isaac Lab above one env. The impulse field
    # already escapes, since collider_basis "pic27" builds a PointBasisSpace.
    # The stress field does not: this setting governs how its warm start
    # transfers, not its basis, which comes from strain_basis "P0" and is
    # cell-wise. "none" leaves the field unallocated so the check skips it,
    # at the cost of the implicit solve starting cold each step. "grid" and
    # "smoothed" are rejected outright with a rebuildable sparse grid.
    #
    # Set here rather than overridden: Hydra reads a bare none as null.
    # "none" zeroes stress_field every step (solver_implicit_mpm.py:
    # `elif not self._stress_warmstart: scratch.stress_field.dof_values.zero_()`)
    # while the collider impulse_field is warm-started unconditionally. So the
    # contact dual variable is converged and the SOIL dual variable is
    # cold-started at 100 Hz -- the worst possible initial guess for a static
    # field whose answer is identical every step, and the structural reason the
    # harvested force reads correct while the bulk does not.
    mpm_warmstart_mode: str = "particles"

    # Early termination has to be off with separate_worlds off: a masked
    # reset cannot clear grid-backed warm starts on a shared grid, so the
    # solver requires every world to reset on the same step. With episodes
    # running their full length that holds, since all envs start together and
    # time out together. The shape_done and failed flags are still raised for
    # the reward and the logs, they just stop ending the episode.
    #
    # It also removes a reward artifact rather than only working around the
    # solver: ending on success means the policy never sees what follows a
    # finished cut, so it never learns to stop cutting. Running to the timeout
    # leaves overcut to charge for digging past the plane.
    terminate_early: bool = False

    # True, and it is not optional for training. Newton refuses a masked reset
    # against grid-backed warm starts on a multi-world model:
    #
    #   RuntimeError: Masked reset cannot selectively clear grid-backed warm
    #   starts when Config.separate_worlds=False for a multi-world model; set
    #   separate_worlds=True or reset all worlds.
    #
    # _validate_reset_warmstart_fields lets a field through only when the mask
    # is None or its basis is a PointBasisSpace. ws_stress_field rides on
    # strain_basis pic8 and is skipped; ws_impulse_field rides on
    # collider_basis Q1 and is not. And the test is on the mask EXISTING, not
    # on it being selective, so every Isaac Lab reset trips it -- the failure
    # lands on the very first env.reset() at 2 envs or more.
    #
    # Nothing else is open. collider_basis to a pic* basis makes
    # activation_distance 0.0 and the machine gets no contact band at all,
    # which was the original bug. One env avoids the check but throughput goes
    # with env count, so a 410k-sample budget is about 91 hours.
    #
    # This field read False on a measurement -- 2 envs failing the rebuild with
    # per-world partitioning on, 4 running clean with it off -- taken at 4x
    # soil density, six vanes and 1e15 stiffness, on a bed being torn apart by
    # a solver that has since been fixed in six ways. It needs re-measuring
    # against the bed as it is now, and the failure mode is loud (LEAF, LOWER
    # and UPPER capacity exceeded), not silent.
    mpm_separate_worlds: bool = True

    # --- regolith ---
    #
    # The yield surface is
    #     tau_max(p) = yield_stress + friction * (p - p_min)
    # so friction is ~tan(phi) and soil_cohesion is cohesion in Pa.
    #
    # At zero cohesion the regolith behaves as dry sand and shears off the lip;
    # a few hundred Pa and the cut travels as a clod the lifters carry round.
    # Lunar simulants sit around 0.1-1 kPa.
    # Surface regolith, from the Apollo in-situ estimates. Density runs
    # 1300-1500 kg/m3 at the surface and reaches 1800-1900 below about 0.3 m,
    # so 1800 was compacted regolith, not surface. Friction angle is 30-50 deg,
    # and cohesion 0.1-1 kPa -- 0 Pa is dry sand, and a cohesionless soil also
    # cannot hold the bed's faces, which is what the retaining plates are for.
    # DELIBERATE FUDGE, not a regolith value. Real surface regolith is about
    # 1500 kg/m3 and that is what luna_hifi_tasks/planner/site.py still
    # models, because the planner describes the real machine on the real Moon.
    #
    # Measured in this solver, at lunar gravity, the bed's bearing capacity is
    # 1.71 kPa where c*Nc + 0.5*gamma*B*Ngamma predicts 64 kPa, and the
    # machine needs 6.5 kPa to sit on the surface rather than sinking 22 cm.
    # The whole of that 1.71 kPa is the self-weight term: halving gravity
    # changed the capacity by 1.90 against a gravity ratio of 2.00, and
    # raising yield_stress tenfold, from 500 to 5000 Pa, moved the sinkage
    # 22.6 cm to 22.1 cm. The cohesion term that should supply the
    # gravity-independent 37.8 kPa does nothing, which contradicts the solver
    # source: shear_yield is s_max + 0.686*p with s_max taken straight from
    # yield_stress, hardening is off so its multiplier is 1, and
    # young_modulus defaults to 1e15 so this is not elastic settlement.
    #
    # Lunar regolith bulk density at the surface. It was 6000, a deliberate
    # 4x, and the note here said so: it bought a trainable environment "while
    # the material model is still wrong", with every number it produced
    # suspect "until that is fixed". It is fixed. The machine now parks on the
    # bed and settles 1 mm with the soil carrying 813 N of its 797 N weight,
    # so the bed no longer needs to be four times too heavy to hold it up.
    #
    # The fudge is not neutral now, it actively blocks the dig. Confining
    # pressure is rho*g*h and the yield surface is 500 + 0.84*p, so 4x the
    # density is roughly 2.5x the shear strength: geostatic at 25 cm goes
    # 2430 -> 608 Pa and the yield there 2541 -> 1011 Pa. Three dig runs ended
    # with the drums reading n=0 contact nodes because the arm saturates
    # pressing into a bed far stronger than the moon's.
    #
    # target_load_kg follows from this through BORE_VOLUME * soil_density, so
    # it moves 160 -> 40 kg on its own, which is the figure the comment on
    # that field was already written against.
    soil_density: float = 1500.0
    soil_friction: float = 0.84              # tan(40 deg)
    # 150, down from 500. Near the surface the confining pressure is tiny --
    # 61 Pa at 2.5 cm once density came back to 1500 -- so the yield surface
    # 500 + 0.84*p is almost all cohesion there, and cohesion is what decides
    # both how far the wheels settle and how hard the vanes have to work.
    #
    # The bind it breaks: the smallest cut this bed can represent is one
    # particle spacing, 40 mm, while the wheels settled 1 mm. To dig, the
    # drums must sit below the wheels' contact plane by the cut depth, so a
    # 40 mm cut puts them 40 mm under wheels that move 1 mm, and the drums
    # take the whole machine. That is why every dig run reported all four
    # wheels at n=0 whatever the vane count, spin direction, shroud aim or arm
    # effort: the cut depth and the wheel sinkage are the same quantity and
    # they were a factor of 40 apart.
    #
    # The vanes are not short of reach -- the 100 degree inlet leaves the
    # shroud's lowest material 0.136 m below the drum axis against the tips'
    # 0.185, so they clear it by 49 mm, and the front shroud carried |net|
    # 0.049 while the drums carried 5.7 and 6.2 N.s. They cut. The reaction
    # just came to 711 N against a 797 N machine.
    #
    # 150 Pa sits inside the Apollo-derived range for loose surface regolith,
    # 0.1 to 1.8 kPa, so it is a layer being modelled rather than a number
    # being tuned. The cost to watch is that this is also the wheels' support:
    # parked sinkage was 1 mm and will grow, which is the point, but if it
    # runs to tens of centimetres the bearing capacity has gone with it.
    soil_cohesion: float = 150.0             # yield_stress, Pa
    soil_yield_pressure: float = 1.0e12      # Pa, compression cap
    # Elastic stiffness. Newton registers this with a default of 1.0e15 and
    # nothing here overrode it, nor does any of Newton's own MPM examples,
    # which is why their two-way example sinks its boxes through half a metre
    # of sand the same way this bed sinks the machine.
    #
    # 1e15 Pa is not a material, it is "effectively rigid", and it is a
    # billion times stiffer than lunar regolith at 10-100 MPa. Two things
    # follow, either of which is fatal. The elastic strain for any soil-scale
    # pressure is unrepresentable: at a bulk modulus of 8.3e14 the geostatic
    # 2430 Pa at this bed's base is a volumetric strain of 2.9e-12 and the
    # 6500 Pa under a wheel is 7.8e-12, against a float32 epsilon of 1.2e-7,
    # and particle_elastic_strain is stored as a float32 mat33. And the
    # rheology solve runs on compliance = 1/E = 1e-15, which leaves the
    # constraint system badly conditioned -- the "slowed convergence" Newton
    # 1.7.0's changelog names, and the reason it added a majorize option to
    # bound the spherical/deviatoric coupling.
    #
    # 1e8 is the STIFF end of measured regolith, and the bed does not need to
    # be at that end. Published values for loose surface material run nearer
    # 1e6..1e7, and the dig runs want the softer figure: the cut reaction has
    # been lifting the machine off its wheels -- 930 N against 797 N of weight
    # at the worst, with all four reading n=0 contact nodes -- and that
    # reaction scales with the stiffness the vanes are working against.
    #
    # Numerically it is still comfortable, and in fact better. The per-step
    # physical strain rho*g*H/E goes 6.1e-6 -> 6.1e-5, which is 30x clear of
    # the 2e-6 tolerance instead of 3x, and 1e-4 volumetric strain is far
    # inside float32's 1.2e-7 epsilon. The failure mode this field's note
    # describes is at 1e15, in the other direction entirely.
    soil_young_modulus: float = 1.0e7        # Pa
    # Maximum solid fraction the material may occupy of a grid node's FREE
    # volume, where free means the node minus whatever a collider occupies:
    #
    #   slack = critical_fraction * (node_volume - collider_volume) - particle_volume
    #
    # clamped at zero. Pressing a wheel in shrinks the free volume, which
    # drives the slack to zero and makes the material resist, and that is the
    # whole mechanism by which a granular bed carries a load. At 0.0 the term
    # is not computed at all -- solver_implicit_mpm zeroes the offset -- so
    # the soil generates no confining pressure, and Drucker-Prager strength is
    # cohesion + friction * pressure, which with no pressure leaves cohesion
    # alone. Measured: the parked machine settles at 22 cm of sinkage, where
    # its 0.46 m2 of contact patch means a bearing capacity of 1.7 kPa against
    # the 64 kPa these parameters should give, and an effective Nc of 3.4
    # against 5.14 for a frictionless cohesive soil. The friction term was
    # contributing nothing.
    #
    # Everything above this line is wrong, and is kept only so the mistake is
    # legible. Two errors, in opposite directions.
    #
    # The arithmetic: this bed does NOT spawn at a solid fraction of 0.893.
    # particles_per_env lays out ceil(2*5.8/0.08) x ceil(2*2.0/0.08) x
    # ceil(2*0.25/0.08) = 145 x 50 x 7 = 50,750, so the spacing is
    # ANISOTROPIC -- 0.04 x 0.04 x 0.035714 -- and a particle occupies
    # 2.9/50,750 = 5.7143e-5 m3, which is what gives the 0.34286 kg the run
    # prints. That is 8.96 particles per 0.08 m node, not 8, and a packing
    # fraction of exactly 1.000000. The offset is
    # max(critical_fraction*(node_volume - collider_volume) - particle_volume,
    # 0), which at fraction 1.0 is max(-0.1*V - 0.9*V_collider, 0) = 0 for
    # every node. So this knob has been INERT at 0.90 exactly as it was at
    # 0.0, which is why the sweep showed no change -- not because the
    # mechanism was saturated.
    #
    # The direction: the term is not an incompressibility constraint. Its own
    # docstring is "Fraction for particles under which the yield surface
    # collapses", and rheology_solver_kernels.py gates
    # `yield_params[1] = 0.0  # disable cohesion if offset > 0 (not compact)`
    # on it. It is a LOOSENESS detector that removes cohesion where the
    # material has dilated. It cannot generate confining pressure and it never
    # could; confining pressure comes from the elastic volumetric branch.
    # So 0.90 does not stiffen the bed -- it arms a term that, wherever soil
    # does loosen enough to fire it, deletes the 500 Pa of cohesion.
    #
    # Back to Newton's default, which switches the term off outright.
    soil_critical_fraction: float = 0.0

    # Derived in __post_init__ and read off the cfg.
    bed_top: float = 0.0
    bed_grid_lower: tuple[float, float] = (0.0, 0.0)
    bed_grid_nx: int = 0
    bed_grid_ny: int = 0
    bed_particles_per_env: int = 0
    drum_capacity_kg: float = DRUM_CAPACITY_KG
    # Work area times the mean commanded depth, derived in __post_init__,
    # which overwrites whatever stands here. The value below is what the
    # current work area and cut depth produce (0.9 x 1.2 m taken down one
    # 0.04 m voxel, 64.8 kg at 1500 kg/m3) so that reading this field off the
    # class gives the right order of magnitude. It previously read 0.22, five
    # times the truth, which is an easy way to misconvert a logged reward
    # back into kilograms.
    cut_volume_ref: float = 0.0432

    # The sparse grid is sized from THIS, not from scene.num_envs, as an
    # absolute total over all envs -- so a run with fewer envs still allocates
    # for this many. At 104 that is 50750 x 104 = 5.28M particles and ~25770 MB
    # of grid against a 3090's 24576, which does not fail with a message about
    # env counts: it hangs and then dies in wp_alloc_device_async.
    #
    # play_dig overrides it per run and re-runs __post_init__. The training
    # path does not, so the committed value has to match what training uses.
    # It is a plain class attribute rather than an annotated dataclass field,
    # so a hydra `env.max_num_envs=` override does not reliably reach it --
    # change it here.
    #
    # 4, because separate_worlds=True (required for the masked resets training
    # performs) costs enough memory that 8 envs fails to allocate 4.8 GB and 4
    # runs clean.
    max_num_envs = 4

    # actions: [forward, yaw, boom, drum, shroud] in [-1, 1]. Boom, drum and
    # shroud are one command each, applied to both ends.
    action_space = 5
    observation_space = DIG_OBS.dim      # 169
    state_space = DIG_CRITIC.dim         # 231

    # --- start state ---
    # Jitter spans what navigate hands over: it declares success at
    # goal_dist_tol 0.5 m in any direction and goal_heading_tol_rad 0.35.
    arm_start_angle = -0.20              # drums ~0.44 m up, clear of the bed
    arm_start_jitter = 0.10              # rad
    spawn_yaw_jitter = 0.40              # rad
    spawn_x_jitter = 0.55                # m
    # Was 0.55, which spawned the machine outside bed_margin_y and would fail
    # the episode on its first step. It also put the work area's far edge at
    # 1.15 m against a bed half-width of 1.00, so the commanded cut ran off
    # the bed entirely. 0.30 keeps both inside: 0.30 + 0.60 = 0.90 < 1.00.
    spawn_y_jitter = 0.30                # m

    # Episodes an env keeps its bed before every particle goes back to its
    # spawn cell. The MPM grid is one spawner shared by all envs, so a carried
    # over bed is what makes one env's ground differ from another's: terrain
    # the machine has already worked.
    soil_reset_every: int = 8

    # --- actions ---
    action_smoothing = 0.3
    # Which drum the dig observation is taken from: the one leading, since the
    # rear assembly is a 180-degree copy and the same drum command spins it the
    # way that cuts when travelling the other way. The machine is meant to
    # shuttle, cutting both directions without turning -- turning a skid steer
    # on regolith at a sixth of a g is where it loses traction.
    #
    # The choice is held through a commanded speed smaller than this, so it
    # does not chatter every time the drive crosses zero and jerk the whole
    # scan 180 degrees between steps.
    dig_lead_deadband: float = 0.05      # m/s of commanded drive

    arm_range = ARM_RANGE
    shroud_range = SHROUD_RANGE

    # --- cut command ---
    # The planner hands down a target plane. Here it is sampled per episode: a
    # depth below the surface the drum meets, plus a gradient.
    # One voxel, fixed. Particles sit one per cell at cell centres, so the
    # surface resolves to voxel_size and a thinner cut is not representable:
    # at 0.04 m a 1.5 cm command is 0.38 of a cell. Measured, one pass clears
    # about 40% of a layer over a 2.76 m2 area, patchily -- the mass divided
    # by the area reads as 1.6 cm but no such surface exists. Commanding the
    # layer makes every cell's outcome binary, which the heightmap can see,
    # depth_error can measure and shape_done can judge.
    cut_depth_range: tuple[float, float] = (0.04, 0.04)

    # Rise over run the planner may command, about 17 degrees. Where the
    # machine stalls is learned inside this bound.
    # Zero: a plane cannot be tilted within one voxel of thickness. At 0.30
    # over the old 2.3 m area the target dropped 0.69 m into a 0.25 m bed and
    # the floor clamp flattened whatever was left, so half the episodes were
    # commanding a wedge the machine cannot cut into ground that is not there.
    # Ramps come from the planner stepping successive flat bites.
    cut_gradient_max: float = 0.0
    # Fraction of episodes with a non-zero gradient; the rest are flat cuts.
    cut_ramp_fraction: float = 0.0
    # Lateral gradient, as a fraction of cut_gradient_max. The planner aligns
    # the approach pose with the ramp, leaving a small residual the policy
    # answers with yaw.
    cut_lateral_fraction: float = 0.3

    # The ground this cut is responsible for, anchored where the drum first
    # meets soil and extending forward along the approach heading. Fixed in the
    # world for the episode, and progress is measured over it.
    # 0.60 ahead plus 0.30 behind is 0.90 m, which with the 1.2 m swath is
    # 1.08 m2: one voxel layer off it is 65 kg, and the drum is throughput
    # limited at 4 to 7 kg/s regardless of depth or speed, so that is about
    # 13 s of cutting inside a 20 s episode. At 2.0 m the commanded layer was
    # 166 kg, three to four passes, and could never be finished in one.
    work_area_length: float = 0.60           # m ahead of the anchor
    work_area_behind: float = 0.3            # m behind it
    work_area_width: float = 1.2             # m, a little over the 1.0 m swath

    # Residual volume, as a fraction of cut_volume_ref, at or below which the
    # commanded shape counts as achieved and the episode ends.
    # 0.30 of the 675 cells in the layer, so 202 may survive. The target is
    # one voxel down and a cell is either cleared or not, so there is no
    # vertical near-miss and the tolerance is entirely about coverage. The
    # first sweep clears the bulk cheaply and the stragglers cost
    # disproportionate time, so 0.10 would never have fired at all. Tighten
    # it as a curriculum once the policy holds this.
    shape_success_fraction: float = 0.30

    # --- load sensor ---
    # What the policy reads in place of the particle count: a boom-torque
    # weighing, modelled as a first-order lag, noise proportional to the
    # reading, and a calibration bias held for the episode. The exact figure
    # stays on the critic as drum_fill_mass.
    # Derived in __post_init__ from the bore volume, so it tracks
    # soil_density. Everything the task is scored on is a mass -- the trim,
    # the payload, removal, unheld, load_ref -- so they all scale together
    # with density and the task stays volumetrically identical. Leaving this
    # at a fixed 40 kg while the soil got four times heavier would instead
    # have made one trim five drum-loads instead of one.
    # 40 kg at 1500 kg/m3, which is 0.52 of a drum's swept volume.
    target_load_bore_fraction: float = 40.0 / (BORE_VOLUME * 1500.0)
    target_load_kg: float = 40.0
    fill_sensor_tau: float = 0.3             # s
    # spill reads its own, much slower filter. The load sensor's 0.3 s is
    # right for a boom-torque weighing, but it is 0.83 of a vane period at
    # 6 vanes and 2.9 rad/s, so it passes 19% of the per-vane fill ripple.
    # spill rectifies and sums whatever it is given, so every vane that
    # passed charged for the mass leaving the bore behind it: 150 kg charged
    # over an episode that captured 23. At 2.0 s the ripple is down to 3%
    # and the term moves on sustained loss, which is what it is for.
    spill_filter_tau: float = 2.0            # s
    fill_sensor_noise: float = 0.05          # fraction of the reading
    fill_sensor_abs_kg: float = 0.5
    fill_sensor_bias: float = 0.03           # +- fraction, per episode

    # --- success ---
    # 0.6 of target_load_kg is 24 kg in each drum.
    fill_success_fraction = 0.6
    # "front": the front drum alone. "all": every drum. "mean": the pair
    # averages past the threshold.
    fill_success_mode = "all"

    # --- reward weights ---
    # One payer, on a quantity the sim conserves.
    #
    # Three terms used to pay for one physical event and each was separately
    # cheatable. depth paid for the work area's surface dropping, which a
    # blade push achieves without excavating; fill paid for mass in the drums,
    # which digging anywhere on the bed achieves without touching the work
    # area; success paid for a small residual, which displacement achieves
    # too. Measured: at one iteration the drums held 45.4 kg while only
    # 20.4 kg had left the work area -- payload exceeding removal, i.e. mining
    # off-target. At another, 24.8 kg left the work area and 0.34 kg was kept.
    #
    # So there is one payer and one penalty, on two instantaneous states:
    #     removal = mass gone from the work area since the episode began
    #     unheld  = |removal - payload|, the mass the pass failed to hold
    # removal pays for cutting the commanded area and nothing else, so soil
    # taken from elsewhere on the bed earns zero. unheld charges the gap both
    # ways: displacing the layer leaves payload at zero and cancels the whole
    # removal payment, and burying the drums past what was cut is charged for
    # the excess rather than rewarded for it. The single optimum is payload ==
    # removal -- keep exactly what you cut -- which is the task.
    #
    # Each is a function of state, not an accumulated delta, so an episode's
    # total telescopes to the final value and a mid-pass fluctuation (soil
    # entering the bore before the heightfield registers it) cannot ratchet
    # into free progress.
    #
    # Both normalise by 2 x target_load_kg, so a weight is points per 80 kg.
    # unheld is the larger of the two on purpose: at equal weights pushing the
    # whole layer aside nets exactly zero, and zero is not a deterrent when
    # every alternative also costs energy. At 120 it nets -15. Checked against
    # the five cases -- ideal 72.9, measured capture 49.5, packing 40.4,
    # pushing -14.6, off-target mining -54.0 -- in which order they must fall.
    # Set from scripts/reward_audit.py, which prints per-term episode totals
    # for a stationary and a fresh policy.
    w_removal = 100.0                    # per 2 x target_load_kg cut on-target
    w_unheld = 120.0                     # per 2 x target_load_kg not held
    # Fraction of the mass removed from the work area that must still be in
    # the drums for the shape to count as achieved. Measured capture over a
    # pass is 0.733, and a 70%-cleared cut is 45.4 kg against a pair that
    # carries about 52, so 0.50 is inside what the hardware does and leaves
    # room for the stragglers at the end of a pass costing capture.
    capture_min_fraction: float = 0.50
    # Distance from its env origin at which a particle counts as thrown clear
    # and ends the episode. The bed reaches 3.4 m and envs are 9 m apart.
    particle_escape_m: float = 15.0
    # Matched to w_removal: they are the reward and the penalty for the same
    # surface, and leaving this behind would make overcutting nearly free.
    # 2x w_removal, not matched to it: going below the plane costs twice what
    # coming down to it earns, because another pass can always take more and
    # nothing can put soil back. Also what stops overcutting from paying for
    # itself -- the policy dug past the plane to pack its drums, and at 100
    # the fill it bought was worth more than the overcut it cost.
    w_overcut = 200.0                    # per cut_volume_ref taken below it
    w_spill = 20.0                       # asymmetry on top of a loss of payload
    # 40, which is what the double-paid 20 was actually worth all along. Made
    # explicit now that success pays once.
    w_success = 40.0
    # Charged once on _failed(), never on a timeout, a finished shape or a
    # full drum. A fresh policy's per-step penalties come to about 33 over a
    # full episode.
    w_fail = 20.0
    w_stall = 0.025                      # commanded speed not achieved
    w_drift = 0.01                       # lateral motion, which a yaw also is
    w_upright = 2.0
    w_energy = 1.0e-5
    w_action_rate = 0.0075               # ~12% of a fresh policy's penalties
    w_time = 0.005

    # --- termination ---
    max_tilt_rad = math.radians(60.0)
    max_env_excursion = 6.0

    # --- scan ---
    # Actor only; the critic sees the bed clean. The window is 2 x 1 m at the
    # drum, and scan_invalid sits outside +- scan_clip.
    scan_clip = 1.0
    scan_noise_std = 0.015               # m, at scan_range_ref
    scan_dropout = 0.02                  # probability, at scan_range_ref
    scan_range_ref = 1.0                 # m
    scan_invalid = -2.0

    # The mass and inertia each wheel presents to the MPM solve.
    #
    # Newton builds the soil side of the contact from these. In
    # rasterized_collisions.build_rigidity_operator the body block of
    # IJtm = M^-1 J^T is (cell_volume / body_mass) * I, and the contact solve
    # closes the gap by iterating
    #     delta_lambda = delta_u / (inv_mass_soil + collider_inv_mass)
    # until the relative normal velocity is zero. A heavier body responds less
    # to a given impulse, so the solve has to drive lambda higher to get there
    # and the harvested reaction rises with it. A lighter body gets a smaller
    # reaction. The note this field used to carry had that backwards -- it read
    # a large mass as the soil being asked to hold up too much -- and set the
    # scale to 1.0 on the strength of it.
    #
    # 1.0 is wrong because build_rigidity_operator reads body_mass and
    # body_inv_inertia per body and knows nothing about the joints. A wheel
    # bolted to a 492 kg rover is handed to the soil as a free 14.1 kg disc
    # (WHEEL_MASS + WHEEL_GROUSERS * GROUSER_MASS), so the contact converges to
    # the impulse that arrests a free wheel rather than the one that holds up
    # the machine. Newton exposes mass_scale for exactly this: "scale factor
    # applied to source effective body mass/inertia", effective as against the
    # modelled free-body value, which the same docstring says it leaves alone.
    #
    # 492 kg over four wheels is 123 kg each, and 123 / 14.1 = 8.7. Parked,
    # with two wheels in contact, the harvested impulses came to 66.6 N against
    # 797 N of weight -- a factor of 12 short while still accelerating
    # downward, which is the effective-mass ratio for a two-wheel stance. The
    # tricycle environment, which does not sink, runs 10.0.
    wheel_proxy_mass_scale: float = 8.7

    # The drums hang off the arms instead of carrying the machine, so their
    # effective mass is their own and the free-body value is already right.
    # They get a separate proxy mapping rather than inheriting the wheel scale,
    # which would read as excessive cutting resistance.
    drum_proxy_mass_scale: float = 1.0

    # Under-relaxation on the rigid<->MPM force exchange. NOT damping: it adds
    # no viscosity and removes no energy, and at convergence it cancels out
    # entirely -- _blend_aitken_forces_kernel is
    #     force = force_previous + omega * (force_raw - force_previous)
    # so a converged pass has force_raw == force_previous, residual 0, and the
    # same answer for any omega. It changes only whether the loop gets there.
    #
    # It has to be below 1.0, which is what Newton defaults to, because the
    # exchange is a fixed-point iteration with a gain that is not small. Per
    # pass: the rigid side integrates under the soil's force, so
    # d(delta)/dF = -dt^2/M; the MPM contact is a velocity-level constraint
    # whose impulse response goes through IJtm ~ cell_volume/m_proxy, so it
    # behaves as a spring of stiffness K = d(F)/d(delta) = m_proxy/dt^2. The
    # loop gain is the product:
    #     g = (m_proxy/dt^2)(dt^2/M) = m_proxy / M
    # The dt^2 cancels exactly, so this instability is timestep-independent --
    # a smaller step will not help.
    #
    # Measured, g tracks M = the wheel's own 14.1 kg rather than the 123 kg it
    # carries, i.e. the articulation does not transmit the chassis inertia
    # within a pass. So g is about the mass scale itself: scale 1 gives g = 1
    # and the machine sank 230 mm without ever settling (marginal stability),
    # scale 2 gives g = 2 and it oscillated +-70 mm, scale 8.7 gives g = 8.7
    # and the bed diverged (|F-I| 15.7, p -3.1 MPa, tau/yield 1670, which is
    # impossible by construction and so is proof of divergence).
    #
    # Relaxed, the gain is |1 - omega*(1+g)|, so stability wants
    # omega < 2/(1+g) = 0.21 at g = 8.7, and the error per pass is smallest at
    #     omega = 1/(1+g) = 1/9.7 = 0.10
    # which zeroes it outright in this scalar reading. That is also exactly
    # Newton's proxy_relaxation_min default.
    #
    # The scalar reading is a simplification: four frictional contacts on an
    # articulated machine make the gain a matrix and g its spectral radius, so
    # treat 0.10 as the right order rather than an exact optimum. The test
    # that settles it is whether sinkage comes out the same across a sweep of
    # omega. If it moves with omega the loop is still not converging and the
    # number is numerics, however smooth the trace looks.
    proxy_relaxation: float = 0.1

    # "fixed" holds proxy_relaxation; "aitken" updates it from successive
    # residuals -- _update_aitken_relaxation_kernel computes
    #     omega <- -omega * (r_prev . dr) / (dr . dr)
    # clamped to [proxy_relaxation_min, proxy_relaxation_max], which is
    # Irons-Tuck dynamic relaxation, the standard accelerator for a
    # partitioned coupling whose gain is not known in advance. It needs more
    # than one pass to have any effect, which is what coupler_iterations gives
    # it. Keep "fixed" while sweeping omega, so the sweep means something.
    proxy_relaxation_mode: str = "fixed"

    # How the rigid entry's wheels and drums are presented to the MPM solve.
    # "lagged" feeds back the previous pass's forces, so a body under a steady
    # load is always supported one step late; "staggered" alternates the two
    # solves within the step. The machine sinks the full 25 cm of the bed and
    # rests on the floor while standing still, on soil that now carries its own
    # weight -- 492 kg at 1.62 m/s2 is 6.5 kPa under a wheel against roughly
    # 60 kPa of bearing capacity, so it should settle about a centimetre.
    # Strengthening the soil fourfold moved it by 2 mm, which puts it in the
    # coupling rather than the material.
    # "lagged" or "staggered". Staggered was tried against the sinkage and
    # moved it 0.309 -> 0.314, i.e. not at all, so this stays on the mode the
    # rest of the configuration was tuned against.
    proxy_mode: str = "lagged"

    # Proxy relaxation passes per coupled step: the fixed-point loop that
    # reconciles the force the soil puts on a wheel with how far that wheel
    # has moved into it. The two are mutually dependent and neither solver can
    # answer alone, so the step has to iterate until they agree.
    #
    # It was 1, which does not iterate at all: the rigid side solves using the
    # force left over from the previous step, MPM solves using the resulting
    # motion, and the step ends without checking that they are consistent. For
    # a drum slicing through soil that stale force is a small relative error.
    # For a body at rest it is fatal, because equilibrium is exact and the
    # lagged force is always slightly too small -- always, never too large,
    # since gravity is constant and the correction always arrives late. A
    # small one-signed deficit per step over 600 steps is the 25 cm the
    # machine sinks, and the curve decelerating as it falls is that deficit
    # shrinking with velocity while never finding the balance point.
    #
    # Corroboration that one pass is not the intent: CouplerProxyMappingCfg
    # carries proxy_relaxation, proxy_relaxation_mode with an "aitken" option,
    # and min/max bounds for it. Aitken relaxation is a convergence
    # accelerator for partitioned fixed-point coupling and does nothing
    # whatsoever at a single pass. CouplerAdmmCfg, the sibling, defaults to 5.
    # The 10.0 proxy mass scale this note used to offer as a second suppressed
    # symptom is not one -- see wheel_proxy_mass_scale. It is the effective
    # mass the wheel carries through its joints, which the MPM solver cannot
    # work out for itself and has to be told.
    #
    # Each pass re-runs both solvers, so this multiplies step cost directly.
    coupler_iterations: int = 4

    # The soil_* fields are plain numbers on this cfg; the solver reads an
    # MPMParticleMaterialCfg on the spawner. __post_init__ calls this, and the
    # env re-runs __post_init__ before spawning, so a field set at any point
    # before gym.make reaches the material.
    def apply_soil_material(self) -> None:
        """Copy the soil_* fields onto the MPM material and the derived capacity."""
        mat = self.scene.soil.spawn.material
        values = {
            "density": self.soil_density,
            "friction": self.soil_friction,
            "yield_stress": self.soil_cohesion,
            "yield_pressure": self.soil_yield_pressure,
            "young_modulus": self.soil_young_modulus,
        }
        for name, value in values.items():
            if not hasattr(mat, name):
                raise AttributeError(
                    f"MPMParticleMaterialCfg has no field {name!r} on this Isaac Lab "
                    f"build; it has {sorted(vars(mat))}. Setting it would be a silent "
                    "no-op and the soil would keep its default."
                )
            setattr(mat, name, value)
        self.drum_capacity_kg = BORE_VOLUME * self.soil_density
        self.target_load_kg = self.target_load_bore_fraction * self.drum_capacity_kg

    def soil_material_mismatch(self) -> str | None:
        """Which soil_* fields no longer match the material, if any."""
        mat = self.scene.soil.spawn.material
        bad = [
            f"{n}: cfg {v:g} vs material {getattr(mat, m):g}"
            for n, m, v in (
                ("soil_density", "density", self.soil_density),
                ("soil_friction", "friction", self.soil_friction),
                ("soil_cohesion", "yield_stress", self.soil_cohesion),
                ("soil_yield_pressure", "yield_pressure", self.soil_yield_pressure),
                ("soil_young_modulus", "young_modulus", self.soil_young_modulus),
            )
            if getattr(mat, m) != v
        ]
        return "; ".join(bad) if bad else None

    def boom_command_for_cut(self, cut: float) -> tuple[float, float]:
        """(boom command in [-1, 1], arm angle in rad) that puts the SHROUD
        `cut` metres below the soil surface.

        All heights in world z:

            wheel plane = the bed surface with spawn_on_bed, else 0, the
                          ground plane beside the pile
            drum bottom = wheel plane + pivot - ARM_LEN*sin(angle) - SHROUD_OUT_R
            target      = bed surface - cut

        The angle is solved per scene, since wheel_plane and bed_top differ
        between the on-bed and pile-in-front beds.
        """
        pivot = CHASSIS_Z + MAST_TOP_Z
        wheel_plane = self.bed_top if self.spawn_on_bed else 0.0
        sin_t = (wheel_plane + pivot - SHROUD_OUT_R - (self.bed_top - cut)) / ARM_LEN
        lo, hi = self.arm_range
        angle = min(max(math.asin(min(max(sin_t, -1.0), 1.0)), lo), hi)
        return 2.0 * (angle - lo) / (hi - lo) - 1.0, angle

    def __post_init__(self) -> None:
        # One nominal cut: the whole work area taken down by the mean
        # commanded depth. It is what removal, unheld and overcut divide
        # by, so their weights read as points per nominal cut, and
        # shape_success_fraction as the residual allowed at the end.
        lo, hi = self.cut_depth_range
        self.cut_volume_ref = (
            (self.work_area_length + self.work_area_behind)
            * self.work_area_width
            * 0.5 * (lo + hi)
        )

        # --- resolve the bed and write it into the scene -------------------
        margin = 0.5 * self.voxel_size
        lower = (self.bed_x[0], self.bed_y[0], margin)
        upper = (self.bed_x[1], self.bed_y[1], margin + self.bed_depth)
        length = self.bed_x[1] - self.bed_x[0]
        width = self.bed_y[1] - self.bed_y[0]

        self.bed_top = margin + self.bed_depth
        hm = self.heightmap_margin
        self.bed_grid_lower = (self.bed_x[0] - hm, self.bed_y[0] - hm)
        self.bed_grid_nx = int(math.ceil((length + 2.0 * hm) / self.voxel_size))
        self.bed_grid_ny = int(math.ceil((width + 2.0 * hm) / self.voxel_size))
        self.bed_particles_per_env = particles_per_env(
            lower, upper, self.mpm_voxel_size, self.particles_per_cell
        )
        spawn = self.scene.soil.spawn
        self.apply_soil_material()

        spawn.lower = lower
        spawn.upper = upper
        spawn.voxel_size = self.mpm_voxel_size
        spawn.particles_per_cell = self.particles_per_cell

        slab = self.scene.mpm_ground
        fm = self.mpm_floor_margin
        slab.spawn.size = (length + 2.0 * fm, width + 2.0 * fm, 0.10)
        slab.spawn.collision_props.contact_margin = margin
        # Centred on the bed in both axes.
        slab.init_state.pos = (
            0.5 * (self.bed_x[0] + self.bed_x[1]),
            0.5 * (self.bed_y[0] + self.bed_y[1]),
            -0.05,
        )

        # Retaining plates, just outside each bed face, their tops flush with
        # the soil surface. The x plates run the full width plus both y plates
        # so the corners are closed.
        t = self.bed_wall_thickness
        h = self.bed_top
        cx = 0.5 * (self.bed_x[0] + self.bed_x[1])
        cy = 0.5 * (self.bed_y[0] + self.bed_y[1])
        parked = not self.bed_walls
        for wall, size, pos in (
            (self.scene.bed_wall_xlo, (t, width + 2.0 * t, h),
             (self.bed_x[0] - 0.5 * t, cy, 0.5 * h)),
            (self.scene.bed_wall_xhi, (t, width + 2.0 * t, h),
             (self.bed_x[1] + 0.5 * t, cy, 0.5 * h)),
            (self.scene.bed_wall_ylo, (length, t, h),
             (cx, self.bed_y[0] - 0.5 * t, 0.5 * h)),
            (self.scene.bed_wall_yhi, (length, t, h),
             (cx, self.bed_y[1] + 0.5 * t, 0.5 * h)),
        ):
            wall.spawn.size = (0.01, 0.01, 0.01) if parked else size
            wall.spawn.collision_props.contact_margin = margin
            wall.init_state.pos = (0.0, 0.0, -10.0) if parked else pos

        # Rails under the wheel tracks. They run the full bed length, stand
        # on the floor slab and their tops finish flush with the soil surface,
        # so a wheel rides at exactly grade; the spawn height already clears
        # that surface by 2 cm, which becomes the wheel settling onto the rail
        # rather than starting interpenetrated.
        #
        # The inner face is set by how far the cutting end reaches and the
        # outer face is taken out to the bed wall. A tyre spans 0.475 to 0.675
        # and bears on whatever part of that is over rail; the width beyond it
        # is there so a wheel drifting outboard stays carried. Drifting inboard
        # it runs off after 72 mm, and that is the limit of the approach: the
        # drift reward term, not the geometry, is what keeps the machine
        # straight, and a pressure-sinkage wheel model is the version that
        # would survive a heading change.
        # Follow wheel_rails whenever the regex is unset or is one of the two
        # canonical sets. Testing this against a re-run matters: the env runs
        # __post_init__ again before spawning and the play script flips
        # wheel_rails between the two, so deriving only from None would see
        # its own previous answer the second time round and reject it.
        if self.soil_contact_regex in (
            None,
            DRUM_CONTACT_BODIES_REGEX,
            SOIL_CONTACT_BODIES_REGEX,
        ):
            self.soil_contact_regex = (
                DRUM_CONTACT_BODIES_REGEX
                if self.wheel_rails
                else SOIL_CONTACT_BODIES_REGEX
            )
        elif not self.wheel_rails and "wheel" not in self.soil_contact_regex:
            raise ValueError(
                "wheel_rails is False and soil_contact_regex does not couple the "
                "wheels, so they would rest on nothing and the machine would drop "
                "to the bed floor. Leave soil_contact_regex at None to have it "
                "follow wheel_rails, or name the wheels in it explicitly."
            )

        cut_half_y = ROTOR_HALF_LEN + SHROUD_END_T
        rail_inner = cut_half_y + self.wheel_rail_clearance
        # Two margins in from the wall's inner face, so the rail's own
        # margin does not sit tangent to it.
        rail_outer = self.bed_y[1] - 2.0 * margin
        if self.wheel_rails and rail_inner - margin <= cut_half_y + margin:
            raise ValueError(
                f"wheel_rail_clearance {self.wheel_rail_clearance} does not clear "
                f"two collider margins of {margin:.3f}: the cutting end reaches "
                f"{cut_half_y:.3f} and a rail face at {rail_inner:.3f} would be "
                f"in contact with it at depth"
            )
        if self.wheel_rails and rail_inner >= rail_outer:
            raise ValueError(
                f"wheel_rail_clearance {self.wheel_rail_clearance} leaves no rail: "
                f"inner face {rail_inner:.3f} is at or past the outer face "
                f"{rail_outer:.3f}"
            )
        rail_w = rail_outer - rail_inner
        rail_cy = 0.5 * (rail_inner + rail_outer)
        railed = self.wheel_rails
        for rail, sign in (
            (self.scene.wheel_rail_ylo, -1.0),
            (self.scene.wheel_rail_yhi, 1.0),
        ):
            rail.spawn.size = (
                (length, rail_w, self.bed_top) if railed else (0.01, 0.01, 0.01)
            )
            rail.spawn.collision_props.contact_margin = margin
            rail.init_state.pos = (
                (cx, sign * rail_cy, 0.5 * self.bed_top)
                if railed
                else (0.0, 0.0, -10.0)
            )

        # Wheels rest on z = 0 in the shared asset cfg: the bed surface with
        # spawn_on_bed, the ground plane without it.
        z = SPAWN_Z + (self.bed_top if self.spawn_on_bed else 0.0)
        self.scene.excavator.init_state.pos = (0.0, 0.0, z)

        total_particles = self.bed_particles_per_env * self.max_num_envs
        active = _next_pow2(int(self.grid_cap_multiplier * total_particles))
        # A leaf is a block of 8^3 = 512 cells, so >> 7 is four times the
        # minimum of one leaf per 512. Capacity follows the grid's spatial
        # spread, which grows past the bed as soil scatters; the floors hold
        # 16384 leaf blocks, about 400 MB.
        leaf = max(active >> self.grid_leaf_shift, 16384)
        lower = max(leaf >> 3, 4096)
        upper = max(lower >> 3, self.grid_upper_min)
        print(
            f"[excavate] bed {length:.1f} x {width:.1f} x {self.bed_depth:.2f} m at "
            f"{self.voxel_size:.3f} m heightmap / {self.mpm_voxel_size:.3f} m solver grid "
            f"({self.particles_per_cell**3:.0f} particles/cell) "
            f"-> {self.bed_particles_per_env} particles/env "
            f"x {self.max_num_envs} max envs = {total_particles} particles"
        )
        print(
            f"[excavate] sparse grid active={active} leaf={leaf} "
            f"lower={lower} upper={upper}"
        )
        print(
            f"[excavate] rigid solver njmax={self.rigid_njmax} "
            f"nconmax={self.rigid_nconmax}"
        )
        if self.wheel_rails:
            print(
                f"[excavate]   wheels on rigid rails: top z={self.bed_top:.3f}, "
                f"|y| {rail_inner:.3f}-{rail_outer:.3f} ({rail_w:.3f} wide); "
                f"drum swath +/-{ROTOR_HALF_LEN:.3f}, margin {margin:.3f}"
            )
        else:
            print(
                "[excavate]   wheels on soil (no rails): expect the machine to sink"
            )
        print(
            f"[excavate]   soil coupling: "
            f"{'drums and shrouds' if self.wheel_rails else 'wheels, drums and shrouds'}"
        )
        print(
            f"[excavate]   {self.grid_cap_multiplier:.0f} cells/particle, "
            f"{1 << self.grid_leaf_shift} cells/leaf, "
            f"~{leaf * 512 * 48 / 1e6:.0f} MB of grid"
        )

        # Newton validates these too, but only after the ~95 s scene build,
        # and each run is ten minutes.
        # _GaussSeidelSolver.solve_granularity is 25 and the host loop is
        # range(max_iterations // solve_granularity), so 24 runs ZERO sweeps
        # and reports success. It cost a day once; it does not get to again.
        if self.mpm_max_iterations < 25:
            raise ValueError(
                f"mpm_max_iterations is {self.mpm_max_iterations}, under the "
                f"Gauss-Seidel solve_granularity of 25. The host loop is "
                f"range(max_iterations // solve_granularity), so this runs "
                f"ZERO sweeps and reports convergence. Use a multiple of 25."
            )
        if self.proxy_relaxation_mode not in ("fixed", "aitken"):
            raise ValueError(
                f"proxy_relaxation_mode must be 'fixed' or 'aitken', got "
                f"{self.proxy_relaxation_mode!r}"
            )
        if not 0.0 <= self.proxy_relaxation <= 1.0:
            raise ValueError(
                f"proxy_relaxation must be in [0, 1], got {self.proxy_relaxation!r}. "
                f"0 is one-way coupling (no feedback at all); 1 is Newton's "
                f"undamped default, which the gain g~{self.wheel_proxy_mass_scale:g} "
                f"needs to be below {2.0 / (1.0 + self.wheel_proxy_mass_scale):.3f}"
            )

        # The wheels and the drums reach the MPM solve as separate proxy
        # mappings so each can carry its own effective mass; see
        # wheel_proxy_mass_scale. The two body sets are disjoint and their
        # union is SOIL_CONTACT_BODIES_REGEX, so nothing is mapped twice. A
        # soil_contact_regex somebody has overridden stays one mapping,
        # because its body set is not ours to split.
        def _soil_proxy(bodies: str, mass_scale: float, pipeline) -> CouplerProxyMappingCfg:
            return CouplerProxyMappingCfg(
                source=RIGID_ENTRY,
                destination=MPM_ENTRY,
                bodies=[bodies],
                mode=self.proxy_mode,
                mass_scale=mass_scale,
                proxy_relaxation=self.proxy_relaxation,
                proxy_relaxation_mode=self.proxy_relaxation_mode,
                collision_pipeline=pipeline,
            )

        # Exactly one mapping per direction may carry the pipeline.
        # _build_proxy_collision_configs keys _ProxyCollisionConfig by
        # (source, destination) and rejects a second mapping in the same pair
        # whose factory is not the *same object*
        # ("Proxy collision_pipeline and collide_interval must match for all
        # proxies from 'rover' to 'soil'"), and isaaclab_contrib wraps each cfg
        # in its own functools.partial, so two cfgs can never pass that test
        # however identical their contents. The one config it does build is
        # looked up per direction, not per mapping, so it serves both.
        #
        # It has to be a pipeline and not None. The default is a factory giving
        # the proxies their own; None instead "passes shared outer contacts to
        # the destination", and the outer one is built for the rigid entry.
        # Parked, with the wheels commanded to zero and the arm stowed, the
        # machine falls through the whole 25 cm bed onto the floor and never
        # finds equilibrium, while the drums still scoop soil -- the MPM solve
        # resolves its own colliders at grid level regardless of this. Wheels
        # getting no contact and drums still loading is what a missing proxy
        # pipeline looks like.
        pipeline = NewtonCollisionPipelineCfg()

        if self.soil_contact_regex == SOIL_CONTACT_BODIES_REGEX:
            proxy_mappings = [
                _soil_proxy(WHEEL_CONTACT_BODIES_REGEX, self.wheel_proxy_mass_scale, pipeline),
                _soil_proxy(DRUM_CONTACT_BODIES_REGEX, self.drum_proxy_mass_scale, None),
            ]
        elif self.soil_contact_regex == DRUM_CONTACT_BODIES_REGEX:
            proxy_mappings = [
                _soil_proxy(DRUM_CONTACT_BODIES_REGEX, self.drum_proxy_mass_scale, pipeline),
            ]
        else:
            proxy_mappings = [
                _soil_proxy(self.soil_contact_regex, self.wheel_proxy_mass_scale, pipeline),
            ]

        self.sim.physics = NewtonCfg(
            solver_cfg=CouplerProxyCfg(
                entries=[
                    CouplerEntryCfg(
                        name=RIGID_ENTRY,
                        solver_cfg=MJWarpSolverCfg(
                            use_mujoco_contacts=False,
                            integrator="implicitfast",
                            njmax=self.rigid_njmax,
                            nconmax=self.rigid_nconmax,
                        ),
                        # The rails belong to the rover's own solver and to
                        # no other: they are deliberately absent from the MPM
                        # entry below, which sees only the bodies named on it,
                        # so the soil passes through them and keeps its full
                        # extent while the wheels ride on them. Two boxes, two
                        # geoms, no meaningful cost to the contact budgets.
                        bodies=[
                            EXCAVATOR_PRIM_REGEX,
                            r"/World/envs/env_.*/WheelRail.*",
                        ],
                        include_static_shapes=True,
                        substeps=2,
                    ),
                    CouplerEntryCfg(
                        name=MPM_ENTRY,
                        solver_cfg=MPMSolverCfg(
                            # The solver's background grid only. Particle
                            # spacing comes from the spawn lattice and the
                            # collider margin from MPM_COLLIDER_MARGIN.
                            voxel_size=self.mpm_voxel_size,
                            grid_type="sparse",
                            grid_padding=0,
                            critical_fraction=self.soil_critical_fraction,
                            # pic8: a PARTICLE strain basis, chosen on
                            # measurement. Bed alone under gravity, 2 s, no
                            # rigid body:
                            #
                            #  basis  settle mm   mean KE J/kg  slope    R2   sxx/szz
                            #   P0        33.64      7.830e-08  0.379  0.476   +0.004
                            #   Q1d    -2419.05      7.300e-01  0.012  0.002  exploded
                            #   Q1d    -8562.44      1.287e+00  0.000  0.015  exploded (250 it)
                            #   pic8      22.52      7.829e-14  0.681  0.640   +0.637
                            #
                            # sxx/szz is the number that matters. A bed in a
                            # walled box cannot expand laterally, so it must
                            # develop K0 = 0.429 (elastic, zero lateral strain)
                            # or 0.357 (Jaky at phi = atan(0.84) = 40.0 deg).
                            # Under P0 it is +0.004, mixed sign, no trend --
                            # there is no confining pressure at all, and since
                            # Drucker-Prager strength is cohesion + friction*p,
                            # that leaves friction*p at 65-410 Pa against 500 Pa
                            # of cohesion. The soil was a ~500 Pa cohesive mud
                            # with a bearing capacity of Nc*c = 2.57 kPa,
                            # against 2.85-6.5 kPa under a wheel. Under pic8 the
                            # middle bins read 0.427 and 0.347, straddling the
                            # requirement, and friction*p reaches 1026-1258 Pa
                            # -- the frictional half of the law finally carries
                            # the material. Mean particle kinetic energy falls
                            # six orders of magnitude, so the bed is at rest
                            # rather than creeping.
                            #
                            # Why P0 fails: P0 stress on Q1 velocity is
                            # single-point reduced integration of the trilinear
                            # hex, 24 velocity DOFs against 6 strain DOFs per
                            # cell. The assembled operator carries 55/63/79/
                            # 101/117/149 spurious non-rigid zero-energy modes
                            # on 5x5x3 .. 16x16x3 slabs; pic8 and Q1d measure 0.
                            # Those modes deform the soil while generating no
                            # stress, which is how a bed compacts without
                            # developing pressure.
                            #
                            # Why not Q1d, which also has 0 modes: it detonates
                            # -- the surface launches 2.4 m upward at 24
                            # iterations and 8.6 m at 250, so more solving
                            # injects more energy and starvation is ruled out.
                            # It is the only basis here that activates the
                            # nodes_per_element > 1 eigenbasis rotation branch
                            # (solver_implicit_mpm.py:3380, and the
                            # rotate_matrix_rows / rotate_vectors /
                            # inverse_rotate_vectors kernels appear in its log
                            # and nowhere else), which is dead code under P0 and
                            # so unexercised upstream. Treat that as an upstream
                            # bug, not a tuning problem.
                            #
                            # Not a complete fix: 22.5 mm of settlement is still
                            # ~8000x the 0.0028 mm a 1-D elastic bed at E=1e8
                            # should give, and the slope ratio is 0.681 rather
                            # than 1.0. But the bed is quiet, confined, and
                            # frictional, which P0's never was.
                            #
                            # The 8 in pic8 was wrong: it is a hard cap on
                            # nodes per cell, and this bed starts AT it.
                            #
                            # "picN" resolves to PointBasisSpace(pic,
                            # max_nodes_per_element=N). The strain DOF array is
                            # sized for every particle, but warp fem's
                            # UnstructuredPointTopology.element_node_count
                            # returns min(N, particles_in_cell) with no warning
                            # or flag. A particle past the Nth in its cell has a
                            # strain DOF that no element references: it is never
                            # interpolated into the stress field, never
                            # constrained by the divergence form, and never sees
                            # the yield surface. It carries nothing and resists
                            # nothing, while its elastic deformation gradient
                            # still accumulates the skew part of the velocity
                            # gradient -- pure rotation, no stretch.
                            #
                            # Counted on this exact lattice (145 x 50 x 7,
                            # spacing 0.04 x 0.04 x 0.035714, binned into 0.08
                            # solver cells), particles per cell at spawn:
                            #   8 -> 5184 cells, 4 -> 2232, 2 -> 174, 1 -> 2
                            # Maximum 8, against a cap of 8. ZERO headroom. Any
                            # vertical compaction at all starts dropping
                            # particles, and the loss lands exactly under the
                            # footing where the soil compacts most -- which is
                            # the soil that has to supply the confinement a
                            # bearing mechanism needs:
                            #    8% compaction -> max 12/cell -> 13.6% dropped
                            #   30%            -> max 16      -> 27.2%
                            #   42% (a 106 mm rut in a 250 mm bed) -> 40.9%
                            #
                            # 16 loses nothing anywhere in that sweep, and was
                            # tried: it raises CUDA error 700, an illegal memory
                            # access, during the run. So the cap cannot simply be
                            # raised. warp fem's
                            # UnstructuredPointTopology.element_node_index
                            # (space/point_basis_space.py) passes
                            # node_index_in_elt straight to the quadrature with
                            # no check against the cell's actual point count,
                            # while element_node_count caps at
                            # min(N, point_count). Any kernel that iterates to
                            # MAX_NODES_PER_ELEMENT rather than to
                            # element_node_count therefore indexes past the
                            # cell's particle list, and raising N from 8 to 16
                            # widens that window. That is the third crash found
                            # in a code path the default configuration never
                            # exercises, after the colour-buffer overflow on any
                            # pic* strain basis and Q1d's eigenbasis rotation
                            # branch.
                            #
                            # So 8 it is, and the truncation is a known cost
                            # rather than something fixed. It is zero at spawn
                            # and stays zero below ~5% compaction; it only bites
                            # where the soil compacts hard, under the footing.
                            # The way out is to stop the bed compacting 42%,
                            # which is the bed-depth and voxel-size question, not
                            # a larger cap.
                            #
                            # pic* is in strain_rebuild_safe, so the rebuildable
                            # sparse grid and the particle warm start are
                            # unaffected.
                            strain_basis="pic8",
                            transfer_scheme="apic",
                            # The rheology solve is a projected Gauss-Seidel
                            # relaxation of the dual (Delassus) system
                            # W sigma + b in N_K, W = C + dt^2 B M^-1 B^T.
                            # A GS sweep transports information exactly one
                            # cell, so equilibrating a load path across a bed
                            # L/h = 5.8/0.08 = 72 cells wide needs >= 72
                            # sweeps; the elastic CFL c*dt/h = 16 cells/step
                            # is a second, independent floor. 24 is below
                            # both. It is also below _GaussSeidelSolver's
                            # solve_granularity of 25, and the host (non-CUDA
                            # -graph) loop is `range(max_iterations //
                            # solve_granularity)`, so on that path 24 runs
                            # ZERO sweeps. Newton's own default is 250; keep
                            # this a multiple of 25.
                            #
                            # Worth knowing about this budget: the loop that
                            # spends it does not measure contact. _run_solver_loop
                            # alternates contact_solver.solve() with
                            # rheology_solver.solve(), then stops on
                            # rheology_solver.eval_residual() alone --
                            # evaluate_strain_residual, which reads delta_stress
                            # over strain nodes and nothing else. The collider
                            # impulse residual is never formed, so a contact that
                            # is still far from equilibrium reports converged and
                            # the loop exits. Raising max_iterations is the only
                            # lever on it from out here; closing it properly means
                            # adding a contact term to the stopping test, which
                            # would be a Newton patch.
                            max_iterations=self.mpm_max_iterations,
                            # ABSOLUTE tolerance on a DIMENSIONLESS per-step
                            # strain: evaluate_strain_residual measures
                            # |D . dsigma| = |tau_new - tau| and
                            # update_condition compares it to tolerance^2 with
                            # no reference to any load or stress scale, while
                            # strain_delta_form carries dt so B v is a strain
                            # increment over one step. The elastic strain that
                            # has to hold this machine up is rho*g*H/K =
                            # 2430/8.33e7 = 2.9e-5, i.e. 3.4x BELOW the old
                            # 1e-4 floor: the entire physical signal sat under
                            # the convergence threshold. Unresolved strain
                            # leaks into the configuration at eps_res*H/dt, so
                            # 1e-4 licensed 1e-4*0.25/0.01 = 2.5 mm/s of creep
                            # by construction -- against a measured 2.58 mm/s.
                            # 1e-5, which is the loosest value that still
                            # resolves the answer. The residual is a
                            # DIMENSIONLESS per-step strain compared against an
                            # absolute threshold, and the elastic strain that
                            # has to hold this machine up is rho*g*H/K =
                            # 2430/8.333e7 = 2.92e-5. So:
                            #   1e-4: 0.29x the signal -- ABOVE it, the solver
                            #         converges by its own test while the whole
                            #         answer sits under the floor
                            #   1e-5: 2.92x below it   -- resolves it
                            #   1e-6: 29.2x below it   -- resolves it and then
                            #         keeps paying for precision nothing reads
                            # At 1e-4 the solve was measured terminating after 6
                            # of the 25 sweeps the graph path offers; at 1e-6 the
                            # count is unknown against a cap of 250, so the
                            # worst case is 40x the rheology work per step. That
                            # cost is real -- a parked 24 s run takes minutes --
                            # and training cannot afford it.
                            tolerance=self.mpm_tolerance,
                            warmstart_mode=self.mpm_warmstart_mode,
                            velocity_basis="Q1",
                            # S2, not pic27, because pic27 switches the
                            # contact activation band off entirely.
                            #
                            # rasterized_collisions.py gates contact on
                            #   bc_active = sdf < activation_distance * voxel_size
                            # with
                            #   activation_distance = 0.0 if degree == 0
                            #                         else 0.5 / degree
                            # and pic27 resolves to a PointBasisSpace, whose
                            # ORDER is 0. So the band is exactly zero and a
                            # particle generates no contact impulse until its
                            # CENTRE is strictly inside the collider. S2 gives
                            # 0.5/2 * 0.08 = 0.020 m; Q1 gives 0.040 m. pic27
                            # is the only choice that gives none.
                            #
                            # Particle centres sit half a spacing above the bed
                            # floor, which at 0.04 m spacing is 0.020 m, so with
                            # no band the bed free-falls that whole distance
                            # before anything resists it. The arithmetic closes
                            # on the measurement to 0.1%: free fall of 0.020 m
                            # at g = 1.62 reaches 0.25456 m/s, one step at
                            # dt = 0.01 travels a further 2.546 mm, total
                            # 22.546 mm -- against 22.520 mm measured on the
                            # bed-alone test. It also explains why the dt sweep
                            # looked rate-free: a geometric gap plus one step
                            # predicts a ratio of 1.216 across an 8x dt range,
                            # measured 1.176.
                            #
                            # And it explains the sinkage the right way round.
                            # mean|dz| / settlement measured 1.016, where
                            # uniform compaction over a fixed base gives
                            # exactly 0.500 and bodily translation gives 1.000.
                            # The bed was not collapsing volumetrically -- it
                            # was falling through its own floor as a block.
                            #
                            # Q1 rather than S2, because S2 cannot be built on
                            # a rebuildable sparse grid here. S2 is quadratic
                            # serendipity: its nodes sit on the 8 vertices AND
                            # the 12 edge midpoints, so
                            # NanogridSpaceTopology.__init__ takes the
                            # need_edge_indices branch (EDGE_NODE_COUNT > 0) and
                            # calls _get_topology_edge_grid, which ends in
                            # _build_rebuildable_incidence_grid(..., 12, ...) and
                            # wp.Volume.allocate_by_voxels failing with
                            # "Failed to create volume" after an out-of-memory in
                            # volume_builder.cu rebuild_count_points. pic27 never
                            # took that path at all, because a PointBasisSpace
                            # has no grid topology to build.
                            #
                            # Q1 is trilinear: EDGE_NODE_COUNT is 0, so no edge
                            # grid is built and the failing call never happens.
                            # It is also the better choice for the original
                            # purpose -- activation_distance is 0.5/degree *
                            # voxel, so Q1 gives 0.040 m against S2's 0.020 m,
                            # twice the contact band, where pic27 gives 0.
                            #
                            # Q1 is in collider_rebuild_safe (pic*, Q1, S2, S3)
                            # so the rebuildable sparse grid is unaffected.
                            collider_basis="Q1",
                            collider_velocity_mode="forward",
                            # Left at "auto" (which resolves to "gs" for a Q1
                            # velocity basis). A ("cr", "gs") Krylov warm start
                            # was briefly set here on the theory that plain
                            # Gauss-Seidel could not transport the load path
                            # across a bed 72 cells wide inside the iteration
                            # budget. Measured instead: the solve reports
                            # "Gauss-Seidel terminated after 6 iterations with
                            # residuals 1.22e-05, 5.46e-05" -- it converges to
                            # its own tolerance in 6 of the 25 sweeps the graph
                            # path offers, so there is no transport deficit to
                            # fix and the extra stage was cost and an untested
                            # code path for nothing.
                            solver="auto",
                            separate_worlds=self.mpm_separate_worlds,
                            # Must be False on a coupled entry.
                            project_outside_colliders=False,
                            # upper <= lower <= leaf <= active
                            max_active_cell_count=active,
                            max_leaf_node_count=leaf,
                            max_lower_node_count=lower,
                            max_upper_node_count=upper,
                        ),
                        bodies=[r"/World/envs/env_.*/(MPMGround|BedWall.*)"],
                        all_particles=True,
                        include_static_shapes=False,
                        include_child_joints=False,
                        substeps=1,
                        in_place=True,
                    ),
                ],
                proxies=proxy_mappings,
                iterations=self.coupler_iterations,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
            num_substeps=1,
            use_cuda_graph=self.use_cuda_graph,
        )
        self.sim.use_newton_actuators = True

    def play_mode(self) -> None:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 2)
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                show_particles=True,
                particle_color=MPM_VISUAL_COLOR,
                eye=(4.0, 4.0, 2.5),
                lookat=(1.0, 0.0, 0.0),
            )
        ]


@configclass
class ExcavatorExcavateSmallEnvCfg(ExcavatorExcavateEnvCfg):
    """A bed that fits a 6 GB laptop: 6,882 particles per env against the full
    bed's 161,001.

    The soil is a pile in front of the machine. The machine sits on the rigid
    ground plane and the front drum alone reaches soil, exercising the particle
    adapter, the coupler mapping and the drum-fill sensor.

    The pile top is 0.125 m up; dig_demo.py solves the arm angle that bites
    from a cut depth.
    """

    # At 0.16 m of depth, 0.09 m of the rotor sits inside the soil column.
    #
    # At the dig angle the drum axis sits at PIVOT_X + ARM_LEN*cos(arm) =
    # 1.65 m and its leading edge at 1.86, so a near edge of 1.80 puts the
    # drum at the start of the bed with the whole of it ahead.
    bed_x: tuple[float, float] = (1.80, 2.70)
    bed_y: tuple[float, float] = (-0.55, 0.55)
    bed_depth: float = 0.16
    spawn_on_bed: bool = False
    # A plate on the near face would stand where the drum comes down.
    bed_walls: bool = False

    # Cohesive enough that a cut clod survives the trip into the bore.
    soil_cohesion: float = 800.0

    # A 1.1 m wide strip, run against by a scripted demo: the jitter is tight
    # enough to keep the drum on the pile and the bed resets every episode.
    spawn_x_jitter: float = 0.15
    spawn_y_jitter: float = 0.0
    spawn_yaw_jitter: float = 0.05
    arm_start_jitter: float = 0.0
    soil_reset_every: int = 1

    max_num_envs = 1
    episode_length_s = 30.0

    def __post_init__(self) -> None:
        self.scene.num_envs = min(self.scene.num_envs, self.max_num_envs)
        self.scene.env_spacing = 8.0
        super().__post_init__()


@configclass
class ExcavatorExcavateMicroEnvCfg(ExcavatorExcavateSmallEnvCfg):
    """Small's pile, squared off and deepened: 8,092 particles against 6,882.

    A one-metre strip starting where the drum's leading edge reaches at the
    dig angle, deep enough to bury 0.09 m of the rotor.
    """

    bed_x: tuple[float, float] = (1.80, 2.80)
    bed_y: tuple[float, float] = (-0.50, 0.50)
    bed_depth: float = 0.21
