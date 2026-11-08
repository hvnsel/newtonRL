# Navigate: drive to a goal pose over generated terrain (worked ground,
# craters, boulders) as a static mesh, under lunar gravity, following a path
# planned over the cell.

from __future__ import annotations

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

from ..excavator_cfg import ARM_STOW_ANGLE, CHASSIS_PRIM, EXCAVATOR_CFG, LUNAR_GRAVITY
from ..mdp.course import COURSE_ROUTE
from ..mdp.observations import (
    NAV_FAR_CELL,
    NAV_FAR_SIZE,
    NAV_NEAR_CELL,
    NAV_NEAR_SIZE,
    NAV_SCAN_CELLS,
    critic_state_spec,
    navigate_obs_spec,
)
from ..terrain_cfg import COURSE_TERRAIN_CFG, EXCAVATION_TERRAINS_CFG

# Height above the chassis the scan rays start from, casting straight down.
SCANNER_HEIGHT = 20.0

NAV_OBS = navigate_obs_spec()
NAV_CRITIC = critic_state_spec(terrain_cells=NAV_SCAN_CELLS)


def _scanner(cell: float, size: tuple[float, float]) -> RayCasterCfg:
    """A yaw-aligned grid of downward rays centred on the chassis."""
    return RayCasterCfg(
        prim_path=CHASSIS_PRIM,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, SCANNER_HEIGHT)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=cell, size=size, ordering="xy"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )


@configclass
class NavigateSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=EXCAVATION_TERRAINS_CFG,
        max_init_terrain_level=2,            # rows are difficulty
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
    far_scanner = _scanner(NAV_FAR_CELL, NAV_FAR_SIZE)      # 27 x 27 at 0.45 m
    near_scanner = _scanner(NAV_NEAR_CELL, NAV_NEAR_SIZE)   # 25 x 25 at 0.20 m


@configclass
class ExcavatorNavigateEnvCfg(DirectRLEnvCfg):
    # 200 Hz physics, 25 Hz policy, 750-step episodes.
    decimation = 8
    episode_length_s = 30.0
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=decimation, gravity=LUNAR_GRAVITY)
    scene: NavigateSceneCfg = NavigateSceneCfg(num_envs=256, env_spacing=8.0, replicate_physics=True)

    # Actions: [forward, yaw] in [-1, 1].
    action_space = 2
    # "policy" group: the actor's whole observation.
    observation_space = NAV_OBS.dim      # 25
    state_space = NAV_CRITIC.dim         # 1361, critic only (includes both scans)

    # --- goals ---
    # Goals come from the flat-patch spots of the env's cell: the spot whose
    # distance from the spawn is nearest a draw from goal_dist_range. The
    # curriculum raises the upper end to goal_dist_max.
    goal_dist_range = (1.5, 3.0)         # m
    goal_dist_max = 18.0                 # m
    # Goal heading: the bearing from spawn to goal plus N(0, goal_yaw_spread),
    # or uniform for goal_yaw_random_frac of episodes.
    goal_yaw_spread = 0.5                # rad
    goal_yaw_random_frac = 0.15

    # --- spawn ---
    # spawn_candidates random spawn spots of the cell at random headings; the
    # first whose ground under the footprint spans less than spawn_max_step is
    # taken, else the most even. spawn_radius limits spots to within that many
    # metres of the cell centre; None is the whole cell.
    spawn_candidates = 4
    spawn_max_step = 0.30                # m
    spawn_radius: float | None = None

    # --- blocked routes ---
    # For goal_blocked_frac of resets the goal is a candidate whose straight
    # route from the spawn steps by more than goal_blocked_step; Episode/blocked
    # logs the share.
    goal_blocked_frac = 0.75
    goal_blocked_step = 0.25             # m

    # --- planned path (mdp/paths.py) ---
    # The cheapest way to the goal over a path_res grid of the cell, re-traced
    # from the chassis every step; the actor sees the points path_lookahead
    # metres along it, and progress and bearing follow it. Within
    # path_clearance of an obstacle (a step over goal_blocked_step between grid
    # neighbours) travel costs path_obstacle_cost per metre; from there the cost
    # falls from path_zone_cost to 1 at path_zone. path_zone must stay within
    # the near scan's reach.
    path_res = 0.25                      # m
    path_lookahead: tuple[float, ...] = (1.0, 2.0)  # m
    path_clearance = 0.70                # m, the machine's half-width is 0.675
    path_zone = 2.0                      # m
    path_zone_cost = 3.0
    path_obstacle_cost = 50.0
    path_max_iters = 256

    # --- collisions ---
    # A step over ram_step between neighbouring near-scan cells inside the
    # machine's outline is a collision; it ends the episode when
    # ram_ends_episode. Episode/rammed logs the share.
    ram_step = 0.25                      # m
    ram_ends_episode = True

    # (spawn x, y, yaw, goal x, y, yaw) in the env frame; None draws them.
    fixed_route: tuple[float, float, float, float, float, float] | None = None

    # --- arrival ---
    # Arrived: within goal_dist_tol and goal_heading_tol_rad of the goal's
    # axis (either end facing along it), slower than goal_speed_tol and
    # goal_yaw_rate_tol, for arrive_hold_steps steps in a row.
    goal_dist_tol = 0.05                 # m
    goal_heading_tol_rad = math.radians(3.0)
    goal_speed_tol = 0.25                # m/s, planar
    goal_yaw_rate_tol = 0.30             # rad/s
    arrive_hold_steps = 25
    # goal_vec_near is the body-frame goal vector / max(dist, goal_near_range).
    goal_near_range = 0.5                # m

    # --- playback ---
    # Steps an episode runs on after arriving, commands latched to zero.
    goal_hold_steps: int = 0
    camera_eye: tuple[float, float, float] = (0.0, -10.0, 28.0)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Draw the goal: a disc on the spot and a bar along its axis.
    goal_marker: bool = False
    # Per-step CSV of pose, goal, errors and speed for this many envs; 0 is off.
    goal_trace_envs: int = 0
    goal_trace_path: str = "goal_trace.csv"

    # --- curriculum ---
    # Success: within curriculum_dist_tol of the goal and slower than
    # curriculum_speed_tol at any point. The goal range widens by
    # curriculum_step when the success rate over the last curriculum_window
    # episodes tops curriculum_success_rate. Per env, a success moves the env
    # up a terrain row; an episode without success that closed less than
    # terrain_demote_fraction of its start distance moves it down.
    curriculum_window = 2048
    curriculum_success_rate = 0.7
    curriculum_step = 1.5                # m
    curriculum_dist_tol = 0.5            # m
    curriculum_speed_tol = 0.25          # m/s
    terrain_demote_fraction = 0.5

    # --- actions ---
    action_smoothing = 0.3               # a_t = (1 - s) a_cmd + s a_{t-1}

    # --- reward weights ---
    w_progress = 5.0                     # per metre of path cost closed
    w_bearing = 0.1
    bearing_fade = (1.0, 2.0)            # m: bearing is 0 inside the first, full beyond the second
    w_goal = 30.0                        # once, on arrival
    # Pose potential (mdp/rewards.pose_potential).
    w_pos_fine = 5.0
    pos_sigma = 0.15                     # m
    w_heading = 5.0
    heading_sigma = 0.10                 # rad
    heading_radius = 2.0                 # m
    w_settle = 1.0                       # per m/s of motion inside the pose tolerances
    w_fail = 20.0                        # once, on tipping, leaving the cell or getting stuck
    w_ram = 50.0                         # once, on a collision that ends the episode
    w_upright = 2.0
    w_slip = 0.006
    w_action_rate = 0.005
    w_energy = 1.3e-5
    w_time = 0.02                        # per step

    # --- termination ---
    max_tilt_rad = math.radians(60.0)
    max_env_excursion = 24.0             # m from the env origin
    # Stuck: beyond goal_near_range of the goal, net movement under
    # stuck_min_move and net turn under stuck_min_turn over stuck_window_steps.
    # 0 steps turns it off.
    stuck_window_steps = 125
    stuck_min_move = 0.5                 # m
    stuck_min_turn = math.radians(15.0)

    arm_hold_angle = ARM_STOW_ANGLE

    # --- scan (critic and collision check) ---
    scan_clip = 1.0                      # m

    def __post_init__(self) -> None:
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=2048,
                nconmax=1024,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            # Global broadphase pair budget against the terrain mesh.
            collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=24_000_000),
            num_substeps=2,
            debug_mode=False,
            default_shape_cfg=NewtonShapeCfg(margin=0.01),
        )
        self.sim.use_newton_actuators = True

    def play_mode(self) -> None:
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 16)
        self.goal_marker = True
        self.stuck_window_steps = 0
        self.ram_ends_episode = False
        self.spawn_radius = 4.2
        self.sim.visualizer_cfgs = [NewtonGLVisualizerCfg(eye=tuple(self.camera_eye), lookat=tuple(self.camera_lookat))]


@configclass
class ExcavatorNavigateCourseEnvCfg(ExcavatorNavigateEnvCfg):
    """mdp/course.py's obstacle course on one cell, for playback. The layout is
    set with env.scene.terrain.terrain_generator.sub_terrains.course.layout=."""

    fixed_route: tuple[float, float, float, float, float, float] | None = COURSE_ROUTE
    goal_hold_steps: int = 1000
    camera_eye: tuple[float, float, float] = (0.5, -5.0, 13.0)
    camera_lookat: tuple[float, float, float] = (0.5, 0.0, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = copy.deepcopy(COURSE_TERRAIN_CFG)
        self.scene.terrain.max_init_terrain_level = 0
