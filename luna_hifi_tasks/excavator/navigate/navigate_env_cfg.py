# navigate_env_cfg.py
#
# Navigation on the rigid tier: procedurally generated, already-worked terrain
# as a static mesh, thousands of environments, no MPM. The observation carries
# two 2-D terrain scans from RayCasters, in the same format the excavate task
# builds from particles (see mdp/terrain.py).
#
# Gravity is lunar. See excavator_cfg.py.

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
from ..mdp.course import COURSE_ROUTE
from ..terrain_cfg import COURSE_TERRAIN_CFG, EXCAVATION_TERRAINS_CFG

# How far above the chassis the rays start, so they start above any terrain
# and cast straight down. Isaac Lab adds it to the ray starts alone, so the
# scan reads chassis z minus hit z; see mdp/terrain.scan_from_raycaster.
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

    # Two windows, rotated by yaw only and centred on the chassis, so the
    # machine sees behind and to either side as far as ahead; see
    # NAV_FAR_NX. The sizes are (NX-1)*cell, since GridPatternCfg puts a ray
    # at both ends of each axis.
    #
    # far: 27 x 27 at 0.45 m, reaching 5.85 m in every direction, 4.1 m past
    # either drum.
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

    # near: 25 x 25 at 0.20 m, reaching 2.4 m in every direction, one cell per
    # wheel width.
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

    # With generator terrain the origins come from the sub-terrain grid.
    # env_spacing is a required field on InteractiveSceneCfg.
    #
    # 256 machines, a quarter of the earlier 1024: the workstation reboots
    # under a full GPU load. The agent's longer rollouts keep each update at
    # 32,768 samples; see agents/rsl_rl_ppo_cfg.py.
    scene: NavigateSceneCfg = NavigateSceneCfg(
        num_envs=256,
        env_spacing=8.0,
        replicate_physics=True,
    )

    # actions: [forward, yaw] in [-1, 1]. No arm or drum actuator is wired.
    action_space = 2
    # The "policy" group: velocities, attitude, the goal and the last action.
    # The actor also gets the two terrain windows, as the image groups
    # "far_scan" and "near_scan", through CNN encoders; the critic reads the
    # clean scan in its own group.
    observation_space = NAV_OBS.dim      # 25
    state_space = NAV_CRITIC.dim         # 1361: both windows clean, and the rest; critic only

    # --- goals ---
    # The initial sampling band, metres. This is the curriculum's bottom rung
    # and it has to be one the policy can actually clear, because
    # _record_outcomes only ever widens the band: there is no demotion, so a
    # band that starts too hard is where training stays. At 3-8 m a fresh
    # policy arrives 8% of the time against a curriculum_success_rate of 0.7,
    # and goal_dist_hi sat at 8.0 for 1800 iterations without moving.
    #
    # 1.5-3.0 m is a real drive -- 1.5 m is half the machine's length, 57
    # steps at 1.1 m/s. The band then widens by curriculum_step on each window
    # that clears the threshold. The low end stays put, which keeps short goals
    # in the mix and stops the policy forgetting them as the ceiling rises.
    #
    # Goals are drawn from the flat-patch spots terrain_cfg.py has the
    # generator find in every cell, within 13 m of the cell centre: the spot
    # whose distance from the spawn is closest to a draw from this band. On
    # hazard cells no spot has a crater or boulder within 2 m.
    goal_dist_range = (1.5, 3.0)
    goal_dist_max = 18.0                 # curriculum ceiling

    # --- spawn ---
    # The machine starts on a random spawn spot anywhere in its cell
    # (SPAWN_PATCHES in terrain_cfg.py), facing a random way, so it starts
    # among the hazards rather than in a clear middle. spawn_candidates spots
    # are drawn and the ground under the machine's footprint ray-cast at
    # each; the first whose ground spans less than spawn_max_step is taken.
    # On generated hazard cells 0.4% of spots had a boulder or crater under
    # the footprint that the flat-patch rings missed between two directions;
    # after this check, none of 3072 spawns at difficulty 0.9-1 had boulder
    # ground under the wheels or deck. The machine is set down on the highest
    # point under its tyres, and the lowest tyre drops 3 cm (median) on
    # hazard cells, 6-8 cm on worked ones, at most 23 cm. Set down at the
    # spot's centre height instead, a tyre sat up to 23 cm into the ground.
    spawn_candidates = 4
    spawn_max_step = 0.30                # m; a boulder stands 0.35+, a crater sinks 0.40+
    # Draw spawns only from spots within this of the cell centre; None is the
    # whole cell. play_mode sets it, so the machine starts inside the frame
    # of the fixed playback camera (see camera_eye).
    spawn_radius: float | None = None

    # --- blocked routes ---
    # For goal_blocked_frac of resets the goal is chosen from those whose
    # straight route from the spawn is blocked: somewhere along it the ground
    # steps by more than goal_blocked_step, a boulder's side or a crater's
    # wall, so the machine has to steer round to get there. Left to chance,
    # half the cells had no obstacles and the hazard cells blocked the
    # straight route in a third to three fifths of episodes, and the policy
    # drove straight into the course's obstacles. Where no candidate is
    # blocked, which on the worked cells is always, the usual goal stands.
    # Episode/blocked logs the share.
    goal_blocked_frac = 0.75
    goal_blocked_step = 0.25             # m, between samples at most 0.25 m apart

    # --- path progress ---
    # Progress pays for closing the distance to the goal the way the machine
    # can drive it, round the craters and boulders, and bearing for speed
    # along that way; see mdp/paths.py. Paid on the straight line, both
    # rewarded driving at the goal right up to an obstacle on the line, and a
    # detour paid less than ramming: the policy drove into the course's
    # obstacles head on. Each cell's ground is read on a path_res grid at the
    # first reset; a grid point is an obstacle where it steps by more than
    # goal_blocked_step to a neighbour, and costs path_obstacle_cost per
    # metre, path_margin_cost within path_margin of one (half the machine's
    # width), 1 elsewhere. Each reset spreads the cost to the new goal over
    # the env's cell. The policy never sees any of it: it still has to find
    # the way round from its scans.
    path_res = 0.25                      # m
    path_margin = 0.7                    # m
    path_margin_cost = 5.0
    path_obstacle_cost = 50.0
    path_max_iters = 256                 # relaxation sweeps; stops early once settled

    # --- collisions ---
    # A crater wall or boulder side inside the machine's outline ends the
    # episode as a failure, w_fail, on that step; see
    # ExcavatorNavigateEnv._rammed_now. The stuck rule took 5 s to notice,
    # and wiggling against the obstacle escaped it. Episode/rammed logs the
    # share either way; play_mode turns the ending off so a clip shows the
    # whole episode.
    ram_step = 0.25                      # m between near-scan cells 0.2 m apart
    ram_ends_episode = True

    # (spawn x, y, yaw, goal x, y, yaw) in the env frame, replacing the random
    # spawn and the drawn goal; None draws them. For constructed playback
    # scenarios, such as the course below.
    fixed_route: tuple[float, float, float, float, float, float] | None = None

    # --- arrival ---
    # Arriving is holding the pose: within goal_dist_tol of the point and
    # goal_heading_tol_rad of the heading, slower than goal_speed_tol and
    # turning slower than goal_yaw_rate_tol, for arrive_hold_steps steps in a
    # row. A step outside any of them restarts the count. The episode ends,
    # and w_goal is paid, on the step the hold completes, so the policy has
    # practised standing on the goal before it is ever allowed to stop there.
    #
    # The pose tolerances are the dig's start state: excavate spawns the
    # machine jittered by up to 5 cm along each axis and 0.05 rad, at rest,
    # and only drives straight from there.
    #
    # The heading is the goal's axis, and either end may face along it. The
    # machine is a 180-degree copy of itself and the dig cuts with whichever
    # drum leads (excavate's dig_lead_deadband), so neither end is the front.
    goal_dist_tol = 0.05                 # m
    goal_heading_tol_rad = math.radians(3.0)
    # Stopped, as far as training noise allows. Every action is sampled with
    # noise, and model_1199 trained at a std near 0.38. In a longitudinal
    # model of this drive -- 994 kg, the wheel servo at its 120 N.m cap, the
    # action smoothing -- a policy holding the goal perfectly under that noise
    # read over 0.10 m/s often enough that only 8% of its holds completed
    # within an episode; at 0.25 m/s every one did. Staying inside 5 cm and
    # 3 deg for a whole second caps the average speed at 5 cm/s and the
    # average turn rate near 0.05 rad/s regardless, so these only catch a
    # machine still driving or spinning. Speed is the planar magnitude, so
    # backing in counts.
    goal_speed_tol = 0.25                # m/s
    goal_yaw_rate_tol = 0.30             # rad/s
    arrive_hold_steps = 25               # 1 s at 25 Hz

    # goal_vec_near is the goal vector divided by max(dist, this): a unit
    # direction beyond it, and 0.1 at 5 cm, where goal_vec_b (divided by
    # goal_dist_max) reads 0.003.
    goal_near_range = 0.5                # m

    # Goal heading, relative to the bearing the machine drove in on, which is
    # how a dig approach arrives. goal_yaw_random_frac of episodes draw a
    # heading uniformly instead, which is where turning on the spot is
    # learned.
    goal_yaw_spread = 0.5                # rad, 1 sigma
    goal_yaw_random_frac = 0.15

    # --- playback ---
    # Keep an episode alive this many steps after it arrives, instead of
    # ending it on the step the hold completes. 0 is off.
    #
    # This is the handover: from that step navigate's commands are latched to
    # zero, which is what it does when it hands the machine to the dig, so the
    # clip shows the machine left standing where navigate put it. 75 steps is
    # three seconds of it.
    #
    # For validation footage only. Leave it at 0 for training, where it would
    # spend that many samples per arrival on a machine with its commands cut.
    goal_hold_steps: int = 0

    # Where the playback camera sits and what it points at.
    #
    # The viewer turns the 12 mm focal_length into a vertical FOV on USD's
    # 15.29 mm aperture: 65 deg, and 97 deg across a 1920 x 1080 frame. From
    # 28 m up and 10 m back, looking at the cell centre, that holds every point
    # of a 16 m disc around the centre in frame: a spawn up to 4.2 m off centre
    # plus a 12 m goal, in any direction. The machine is ~100 px long.
    #
    # The old eye=(13, 13, 9) looked across the ground from the side, and
    # clipped part of even a 13 m disc: the near side fell out of the bottom of
    # the frame.
    #
    # Set these here, not with env.camera_eye= on the command line. Isaac Lab
    # 3.0 runs play_mode() before it applies env.* overrides, and play_mode()
    # is where the viewer cfg is built from these, so an override changes the
    # field after the viewer has already copied it.
    camera_eye: tuple[float, float, float] = (0.0, -10.0, 28.0)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Draw the goal in the viewer: a disc on the goal spot and a bar through it
    # along the goal's axis, so footage shows where the machine is meant to
    # stop and which way it is meant to line up. Visual only; the scanners
    # cast against the terrain mesh. play_mode() turns it on.
    goal_marker: bool = False

    # --- trace ---
    # Per-step CSV of where the machine is, where the goal is, how far off
    # its heading is and how fast it is going, for this many envs. 0 disables
    # it.
    #
    # This exists because nothing renders the goal, so footage cannot answer
    # the questions that matter: how close the machine finished, how well it
    # was lined up, and whether it was still moving. Those are numbers, not
    # pictures.
    goal_trace_envs: int = 0
    goal_trace_path: str = "goal_trace.csv"

    # --- curriculum ---
    # Two independent ladders. Goal distance widens globally on the success
    # rate over a window of finished episodes; terrain difficulty is per-env
    # and moves through TerrainImporter.update_env_origins on every reset.
    # At 256 envs and 128-step rollouts about 60 episodes finish per
    # iteration, so the window spans some 35 iterations.
    curriculum_window = 2048             # episodes averaged for the success rate
    curriculum_success_rate = 0.7        # widen the band above this ...
    curriculum_step = 1.5                # ... by this many metres
    # Both ladders count an episode a success when the machine got within
    # curriculum_dist_tol of the goal and slower than curriculum_speed_tol at
    # any point: navigation, not parking. The held arrival pays w_goal but
    # does not gate the curriculum. If it did and held arrivals plateaued
    # under curriculum_success_rate, the band would stay at 1.5-3 m and the
    # long drives through the hazards would never come.
    curriculum_dist_tol = 0.5            # m
    curriculum_speed_tol = 0.25          # m/s
    # An episode that ended without getting near the goal and closed less
    # than this fraction of its starting distance moves that env down a level.
    terrain_demote_fraction = 0.5

    # --- actions ---
    action_smoothing = 0.3               # EMA: a_t = (1-s) a_cmd + s a_{t-1}

    # --- reward weights ---
    # bearing is scaled by forward speed (see mdp/rewards.bearing_alignment),
    # so its episode ceiling is ~6. It fades out between bearing_fade[1] and
    # bearing_fade[0] metres from the goal: it pays for driving at the point,
    # and on the point that is driving back and forth across it.
    #
    # Set from scripts/reward_audit.py, which prints per-term episode totals
    # over a zero pass and a random walk.
    w_progress = 5.0
    w_bearing = 0.1
    bearing_fade = (1.0, 2.0)            # m: 0 inside the first, full beyond the second
    # Paid once, on the step the hold completes. Arriving also stops the
    # per-step penalties, which come to ~0.037/step: finishing at step 150
    # instead of timing out at 750 is worth ~22 on its own, so the bonus does
    # not have to be large. It does have to stay small enough for the critic
    # to fit, since it lands on a minority of episodes: the return variance it
    # injects goes as the square of this.
    w_goal = 30.0
    # The pose terms, as a potential; see mdp/rewards.pose_potential. The
    # reward is the change in it, so an episode collects only the difference
    # between where it started and where it ended. progress pulls at 5 per
    # metre all the way in; these pull on the last 15 cm and the last few
    # degrees, which progress barely notices. heading_radius keeps
    # w_heading * 2 / heading_radius at w_progress, so backing away from the
    # goal never pays for a bad heading.
    w_pos_fine = 5.0
    pos_sigma = 0.15                     # m
    w_heading = 5.0
    heading_sigma = 0.10                 # rad, ~6 deg
    heading_radius = 2.0                 # m
    # Charged on motion while inside the pose tolerances, in m/s: planar speed
    # plus yaw rate times half the track. The hold already demands stopping;
    # this says so on every step instead of only at the end.
    w_settle = 1.0
    # Charged once on _failed(), never on a timeout or an arrival. A fresh
    # policy's per-step penalties come to about 24 over a full episode.
    w_fail = 20.0
    w_upright = 2.0
    w_slip = 0.006
    w_action_rate = 0.005                # ~12% of a fresh policy's penalties
    w_energy = 1.3e-5
    w_time = 0.02
    # Reads zero on this tier: no arm actuator is wired. An assertion in
    # reward form.
    w_fill_change = 1.0

    # --- termination ---
    max_tilt_rad = math.radians(60.0)
    max_env_excursion = 24.0             # metres from the env origin
    # Stuck: away from the goal (beyond goal_near_range), the machine has to
    # move stuck_min_move or turn stuck_min_turn over any stuck_window_steps
    # in a row, or the episode ends as a failure and w_fail is charged.
    # Pushing into a boulder or a crater wall does neither; before this it
    # cost only w_time, and the policy pushed into one on the course until
    # the clip ran out. Both are net changes across the window: backing off
    # and driving back in is still stuck, and so is steering jitter, which
    # training's action noise supplies on every step and which summed turning
    # had counted as turning. Reorienting on the spot still counts. 0 turns
    # it off, which play_mode does so a clip shows the whole episode.
    stuck_window_steps = 125             # 5 s at 25 Hz
    stuck_min_move = 0.5                 # m
    stuck_min_turn = math.radians(15.0)

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
        self.sim.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                # Fixed per-env buffers; overrunning one is an illegal
                # access. Eight machines over generated terrain ask for
                # njmax 504, and a constraint is a couple of hundred bytes.
                njmax=2048,
                nconmax=1024,
                cone="pyramidal",
                impratio=1,
                integrator="implicitfast",
                use_mujoco_contacts=False,
            ),
            # Broadphase pair budget against the terrain mesh, a global
            # total. A measured run at 1024 machines on the 32 m terrain peaks
            # at 12.9M pairs, about 12,600 per machine: the six grousers on
            # each wheel carry far more triangles into an inflated AABB than a
            # bare cylinder would. A pair is a couple of indices, so 24M is
            # about 384 MB.
            #
            # Overflow drops the pairs past the budget, so contacts go missing
            # rather than the run failing, and the log fills with
            # "Triangle pair buffer overflowed" once per step.
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
        self.sim.visualizer_cfgs = [
            NewtonGLVisualizerCfg(
                eye=tuple(self.camera_eye),
                lookat=tuple(self.camera_lookat),
            )
        ]


@configclass
class ExcavatorNavigateCourseEnvCfg(ExcavatorNavigateEnvCfg):
    """Playback only: mdp/course.py's obstacle course on one cell.

    The machine spawns facing a goal 11 m straight ahead with obstacles on the
    line between, so getting there means steering round them. The layout is
    chosen with env.scene.terrain.terrain_generator.sub_terrains.course.layout=.
    Observations and actions are the navigate task's, so a navigate
    checkpoint plays here unchanged.
    """

    fixed_route: tuple[float, float, float, float, float, float] | None = COURSE_ROUTE
    # Parked on the goal until the episode times out, so a clip of the
    # episode ends on the handover pose rather than a reset.
    goal_hold_steps: int = 1000
    # The course spans x -8 to 9 m and y +-6.5 m, spawn to the tip of the goal
    # bar and round the far end of the gate, so it can be shot closer than the
    # navigate disc: this frames it with 18% to spare and the machine ~210 px
    # long. The lens is as described at camera_eye above.
    camera_eye: tuple[float, float, float] = (0.5, -5.0, 13.0)
    camera_lookat: tuple[float, float, float] = (0.5, 0.0, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = copy.deepcopy(COURSE_TERRAIN_CFG)
        # One row of terrain, so level 0 is the only level.
        self.scene.terrain.max_init_terrain_level = 0
