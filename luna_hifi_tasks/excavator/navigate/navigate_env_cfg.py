# navigate_env_cfg.py
#
# Navigation on the rigid tier: procedurally generated, already-worked terrain
# as a static mesh, thousands of environments, no MPM. The observation carries
# two 2-D terrain scans from RayCasters, in the same format the excavate task
# builds from particles (see mdp/terrain.py).
#
# Gravity is lunar. See excavator_cfg.py.

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)

from ..excavator_cfg import (
    ARM_STOW_ANGLE,
    CHASSIS_PRIM,
    EXCAVATOR_CFG,
    LUNAR_GRAVITY,
)
from ..mdp.observations import (
    NAV_FAR_BIAS,
    NAV_FAR_CELL,
    NAV_FAR_SIZE,
    NAV_NEAR_BIAS,
    NAV_NEAR_CELL,
    NAV_NEAR_SIZE,
    NAV_SCAN_CELLS,
    critic_state_spec,
    navigate_obs_spec,
)
from ..terrain_cfg import EXCAVATION_TERRAINS_CFG

# Height the ray-caster origin sits above the chassis. Rays cast straight down
# from here and the observation subtracts it back out, so the scan reads
# chassis-relative ground height, as Isaac's height_scan does.
SCANNER_HEIGHT = 20.0

NAV_OBS = navigate_obs_spec()
NAV_CRITIC = critic_state_spec(terrain_cells=NAV_SCAN_CELLS)


@configclass
class NavigateSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=EXCAVATION_TERRAINS_CFG,
        # rows are difficulty; start on the gentle ones
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    excavator: ArticulationCfg = EXCAVATOR_CFG

    # Two windows, rotated by yaw only. The sizes are (NX-1)*cell, since
    # GridPatternCfg puts a ray at both ends of each axis.
    #
    # far: 16 x 8 at 0.45 m, 2.40 m forward, reaching x = 5.775, four metres
    # past the front of the machine.
    far_scanner = RayCasterCfg(
        prim_path=CHASSIS_PRIM,
        offset=RayCasterCfg.OffsetCfg(pos=(NAV_FAR_BIAS, 0.0, SCANNER_HEIGHT)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=NAV_FAR_CELL, size=NAV_FAR_SIZE, ordering="xy"
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )

    # near: 12 x 8 at 0.20 m, 1.30 m forward. 1.40 m wide, the wheel track.
    near_scanner = RayCasterCfg(
        prim_path=CHASSIS_PRIM,
        offset=RayCasterCfg.OffsetCfg(pos=(NAV_NEAR_BIAS, 0.0, SCANNER_HEIGHT)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=NAV_NEAR_CELL, size=NAV_NEAR_SIZE, ordering="xy"
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )


@configclass
class ExcavatorNavigateEnvCfg(DirectRLEnvCfg):
    # 200 Hz physics, policy at 25 Hz, 750 steps per episode. Top speed is
    # 1.5 m/s, so 30 s covers a goal_dist_max traverse with margin for turning
    # and slip.
    decimation = 8
    episode_length_s = 30.0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        gravity=LUNAR_GRAVITY,
    )

    # With generator terrain the origins come from the sub-terrain grid, and
    # env_spacing is carried because InteractiveSceneCfg requires it.
    scene: NavigateSceneCfg = NavigateSceneCfg(
        num_envs=1024,
        env_spacing=8.0,
        replicate_physics=True,
    )

    # actions: [forward, yaw] in [-1, 1]. No arm or drum actuator is wired.
    action_space = 2
    observation_space = NAV_OBS.dim      # 247
    state_space = NAV_CRITIC.dim         # 231, critic only

    # --- goals ---
    goal_dist_range = (3.0, 8.0)         # initial sampling band, metres
    goal_dist_max = 18.0                 # curriculum ceiling
    goal_dist_tol = 0.5                  # reached when closer than this ...
    goal_heading_tol_rad = 0.35          # ... and facing within this
    spawn_xy_jitter = 3.0                # random offset inside the terrain cell

    # Goals are held inside the sub-terrain the env was assigned. Cells are
    # 32 m and the machine is 3.78 m long, so the margin leaves it room to
    # turn at the boundary. goal_cell_half is derived in __post_init__ as
    # 0.5 * cell - margin, which is 13 m: far enough that far_scan's 4 m
    # lookahead is choosing a line rather than looking past the goal.
    goal_cell_margin = 3.0
    goal_cell_half = 13.0

    # Goal heading, relative to the bearing the machine drove in on, which is
    # how a dig approach arrives. goal_yaw_random_frac of episodes draw a
    # heading uniformly instead, which is where turning on the spot is
    # learned.
    goal_yaw_spread = 0.5                # rad, 1 sigma
    goal_yaw_random_frac = 0.15

    # --- curriculum ---
    # Two independent ladders. Goal distance widens globally on the success
    # rate over a window of finished episodes; terrain difficulty is per-env
    # and moves through TerrainImporter.update_env_origins on every reset.
    # A reset batch at 1024 envs can be most of the batch at once, so the
    # window spans about two reset waves.
    curriculum_window = 2048             # episodes averaged for the success rate
    curriculum_success_rate = 0.7        # widen the band above this ...
    curriculum_step = 1.5                # ... by this many metres
    # An episode that ended without reaching the goal and closed less than
    # this fraction of its starting distance moves that env down a level.
    terrain_demote_fraction = 0.5

    # --- actions ---
    action_smoothing = 0.3               # EMA: a_t = (1-s) a_cmd + s a_{t-1}

    # --- reward weights ---
    # bearing is scaled by forward speed (see mdp/rewards.bearing_alignment),
    # so its episode ceiling is ~6.
    #
    # Set from scripts/reward_audit.py, which prints per-term episode totals
    # over a zero pass and a random walk.
    w_progress = 5.0
    w_bearing = 0.1
    w_goal = 20.0
    w_upright = 2.0
    w_slip = 0.006
    w_action_rate = 0.05
    w_energy = 1.3e-5
    w_time = 0.02
    # No arm actuator is wired on this tier, so this term reads zero. It is
    # an assertion in reward form.
    w_fill_change = 1.0

    # --- termination ---
    max_tilt_rad = math.radians(60.0)
    max_env_excursion = 24.0             # metres from the env origin

    # --- arms ---
    arm_hold_angle = ARM_STOW_ANGLE

    # --- scan ---
    # Actor only; the critic sees both windows clean. Noise and dropout grow
    # with range, so the far window is degraded harder than the near one, and
    # a dropped cell reads scan_invalid.
    scan_clip = 1.0
    scan_noise_std = 0.02                # m, at scan_range_ref
    scan_dropout = 0.02                  # probability, at scan_range_ref
    scan_range_ref = 4.0                 # m
    scan_invalid = -2.0                  # outside +- scan_clip

    def __post_init__(self) -> None:
        cell = min(self.scene.terrain.terrain_generator.size)
        self.goal_cell_half = 0.5 * cell - self.goal_cell_margin

        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                # Fixed per-env buffers; overrunning one is an illegal
                # access. Driving eight machines over generated terrain asks
                # for njmax 504, and a constraint is a couple of hundred
                # bytes.
                njmax=2048,
                nconmax=1024,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            # Broadphase pair budget against the terrain mesh, a global total.
            # At 1024 machines the wheels alone put ~29k colliders near the
            # mesh, ~50 triangles each in an inflated AABB, so ~1.4M pairs. A
            # pair is a couple of indices, so 10M costs on the order of 160 MB.
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=10_000_000),
            num_substeps=2,
            debug_mode=False,
            default_shape_cfg=NewtonShapeCfg(margin=0.01),
        )
        self.sim.use_newton_actuators = True

    def play_mode(self) -> None:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 16)
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(eye=(6.0, 6.0, 4.0), lookat=(0.0, 0.0, 0.0))
        ]
