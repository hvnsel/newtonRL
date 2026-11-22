# site_env.py
#
# One machine on a large bed of the RFT soil, never reset, digging a pit or a
# trench with the trained policies. The runner puts the env in one of four modes:
#
#   ready  wheels held, the arms and shrouds to the pose a dig episode starts
#          from in training (both arms at arm_start_angle, shrouds 0), drums still.
#   dig    a pass of the excavate policy. begin_pass() puts the work area and
#          the plane where the planner says; the settle, ramp, line hold and
#          actions are the excavate env's, as in training, from that pose.
#   drive  the navigate policy, waypoint by waypoint along a route to a goal
#          pose. nav_obs() is its observation; its [forward, yaw] drive the
#          wheels as in navigate, the arms stowed, the drums still, the inlets up.
#   dump   scripted: wheels held, the front inlet turned down and its drum
#          reversed, its load laid on the bed under it.
#
# The soil steps in every mode, with the wheels on it.
#
# The pit runs along +x from pit_x, pit_lanes lanes wide, each lane the drum's
# width, centred on y = 0; one lane is a trench. The machine works a lane from
# along it, its wheels on the lanes either side (outside the pit beside the
# side lanes), and gets in and out at the pit's west end: it lines up on a
# lane at the yard, STAGE short of the pit, and drives in along the lane. To
# dump it backs out to the yard, goes up to the way round, which runs beside
# the pit on its +y side, and pours on one of PILES piles along it. It spawns
# at the origin facing +x, its drum over the first patch of the middle lane.
#
# The machine has a drum at each end, and navigate brings it to a goal with
# either end forward along the goal heading. So a pass digs the way the
# machine faces along its lane, from that end of the patch, the planner picking
# a cut it can make facing that way (it turns on the spot if there is none),
# and a dump pours wherever the front drum is: outside the way round or between
# it and the pit, the way far enough out that either clears both.
#
# run_site() digs a planner's site (luna_hifi_tasks/planner/site.py), reading
# each patch's height back from the bed after its pass.

from __future__ import annotations

import math

import torch

from isaaclab.utils.configclass import configclass

from ..excavate.excavate_env_cfg import BED_FLOOR_Z, ExcavateRftSceneCfg, ExcavatorExcavateRftEnvCfg
from ..excavate.excavate_rft_env import ExcavatorExcavateRftEnv
from ..excavator import ARM_LEN, PIVOT_X, SHROUD_OUT_R
from ..excavator_cfg import ARM_STOW_ANGLE
from ..mdp import rewards as R
from ..mdp import rft
from ..mdp.observations import navigate_actor_obs, navigate_obs_spec
from ..navigate.navigate_env_cfg import ExcavatorNavigateEnvCfg

REACH = PIVOT_X + ARM_LEN * math.cos(ARM_STOW_ANGLE)   # m the front drum sits ahead of the chassis, stowed
STAGE = 0.6                 # m short of the pit the front drum stops, lined up on its lane, before driving in
PILE_CLEAR = 0.7            # m from the pit's edge to a pile poured between it and the way round
PILES, PILE_STEP = 12, 0.6  # dump stops along the way round, this far apart, from the pit's west end
WAYPOINT_RADIUS = 0.5       # m: a route waypoint counts as passed this close
LOOSE_M, LOOSE_DEG = 0.25, 10.0     # a stop short of this is a miss; the yard and dump stops need no closer
DUMP_MIN_KG = 5.0           # less soil than this in the drum: no trip to the pile
TURN_TRIES = 4              # turns on the spot to face the other way before the run gives up
DRIVE_SECONDS, PASS_SECONDS, DUMP_SECONDS, READY_SECONDS = 90.0, 40.0, 15.0, 3.0


def site_layout(cfg) -> tuple[float, float, float]:
    """The pit's west end, its half width and the way round's y, env-local, m.
    The machine spawns at the origin with its drum, at the arm angle a pass
    starts from, over the middle lane's first anchor."""
    x0 = PIVOT_X + ARM_LEN * math.cos(cfg.arm_start_angle) - cfg.work_area_behind
    half = 0.5 * cfg.pit_lanes * cfg.work_area_width
    return x0, half, half + PILE_CLEAR + SHROUD_OUT_R + REACH


@configclass
class ExcavatorSiteEnvCfg(ExcavatorExcavateRftEnvCfg):
    """One machine on a bed laid out round a pit pit_lanes lanes wide and
    pit_length long (one lane: a trench), the yard, the way round and its piles."""

    scene: ExcavateRftSceneCfg = ExcavateRftSceneCfg(num_envs=1, env_spacing=5.12, replicate_physics=True)
    episode_length_s = 1.0e5
    pit_length: float = 2.4                    # m
    pit_lanes: int = 1                         # across, each the drum's width
    spawn_y_jitter = 0.0
    soil_reset_every: int = 1_000_000
    bed_margin_x: float = 0.5
    bed_margin_y: float = 0.5
    max_env_excursion = 12.0
    # Ruts and the spots it turns on stop sinking 3 cm down: hours of traffic, not one episode.
    soil_compaction: float = 0.03
    # Dumping: the front inlet dump_tilt off straight down, its load laid out at dump_rate.
    dump_tilt: float = 0.0                     # rad
    dump_rate: float = 30.0                    # kg/s

    def __post_init__(self) -> None:
        x0, half, way = site_layout(self)
        # West: the yard, the machine turning there; east: a pass from beyond
        # the pit's end, and the piles; across: the side lanes' outer wheels
        # and the piles outside the way round.
        self.bed_x = (x0 - STAGE - 2.0 * REACH - 1.5, x0 + max(self.pit_length + 3.5, PILE_STEP * (PILES - 1) + 1.5))
        self.bed_y = (-half - 1.8, way + REACH + 1.0)
        super().__post_init__()


def along(points: torch.Tensor, distances) -> torch.Tensor:
    """(len(distances), 2) points those distances along the polyline through
    points (P, 2); past its end, its end."""
    seg = points[1:] - points[:-1]
    cum = torch.cat([points.new_zeros(1), torch.linalg.norm(seg, dim=-1).cumsum(0)])
    out = []
    for d in distances:
        if d >= float(cum[-1]):
            out.append(points[-1])
            continue
        i = int((cum <= d).sum()) - 1
        out.append(points[i] + (d - cum[i]) / (cum[i + 1] - cum[i]).clamp_min(1e-9) * seg[i])
    return torch.stack(out)


class ExcavatorSiteEnv(ExcavatorExcavateRftEnv):
    cfg: ExcavatorSiteEnvCfg

    def __init__(self, cfg: ExcavatorSiteEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        assert self.num_envs == 1, "the site env runs one machine"
        dev = self.device
        self.mode = "dig"
        # The front drum's reach at the arm angle a pass starts from: the
        # machine spawns there, with its drum over the first patch's anchor.
        self.dig_reach = PIVOT_X + ARM_LEN * math.cos(cfg.arm_start_angle)
        self.pit_x, self.half_width, self.way = site_layout(cfg)
        self.lanes = cfg.pit_lanes
        self.yard_x = self.pit_x - STAGE - REACH     # where the chassis lines up on a lane
        self.patch_length = cfg.work_area_behind + cfg.work_area_length
        self._nav = ExcavatorNavigateEnvCfg()
        self._nav_spec = navigate_obs_spec()
        self._nav_actions = torch.zeros(1, 2, device=dev)
        self._route = torch.zeros(1, 2, device=dev)
        self._route_yaw = torch.zeros(1, device=dev)
        self._route_next = 0
        self._goal_xy = torch.zeros(1, 2, device=dev)
        self._goal_yaw = torch.zeros(1, device=dev)
        self._held = 0
        self._dumps = 0
        self._soil_drawn = 0

    def _show_soil(self) -> None:
        """The soil boxes once an env step, for its one rendered frame, not every physics step."""
        self._soil_drawn += 1
        if self._soil_drawn % (1 if self._physics_handles_decimation else self.cfg.decimation) == 0:
            super()._show_soil()

    # --- the runner's commands ---

    def begin_pass(self, anchor_x: float, anchor_y: float, plane_z: float, facing: int) -> float:
        """A dig pass along +x (facing 1) or -x (facing -1), its work area
        anchored at (anchor_x, anchor_y) and its plane at plane_z, env-local: the
        depth below the work area's mean surface, kept inside the trained range.
        Returns that depth."""
        ids = torch.zeros(1, dtype=torch.long, device=self.device)
        o = self.scene.env_origins
        anchor = o[:, :2] + torch.tensor([[anchor_x, anchor_y]], device=self.device)
        yaw = torch.full((1,), 0.0 if facing > 0 else math.pi, device=self.device)
        self._set_work_area(ids, anchor, yaw)
        work = self._work_mask & (self._bed_grid > BED_FLOOR_Z + 0.5 * self.cfg.voxel_size)
        mean = (self._bed_grid * work).sum(dim=(1, 2)) / work.sum(dim=(1, 2)).clamp_min(1)
        lo, hi = self.cfg.cut_depth_range
        depth = (mean - plane_z).clamp(lo, hi)
        arm = torch.full((1,), self.cfg.arm_start_angle, device=self.device)     # ready() put the arms there
        self._begin_pass(ids, anchor, o[:, 2] + mean, yaw, arm, depth, torch.zeros(1, 2, device=self.device))
        self.mode = "dig"
        return float(depth[0])

    def begin_drive(self, route: list[tuple[float, float]], goal_yaw: float) -> None:
        """Drive along env-local waypoints to a goal pose at the last, facing
        along goal_yaw (either way along it, as navigate trains). The policy's
        goal is one waypoint at a time, facing along the leg into it."""
        here = self.robot.data.root_pos_w.torch[0, :2] - self.scene.env_origins[0, :2]
        self._route = torch.cat([here[None], torch.tensor(route, device=self.device)])
        legs = self._route[1:] - self._route[:-1]
        self._route_yaw = torch.atan2(legs[:, 1], legs[:, 0])
        self._route_yaw[-1] = goal_yaw
        self._route_next = 1
        self._goal_xy = self.scene.env_origins[:, :2] + self._route[-1]
        self._goal_yaw[:] = goal_yaw
        self._nav_actions.zero_()
        self._held = 0
        self.mode = "drive"

    def begin_dump(self) -> None:
        self._dumps += 1
        self.mode = "dump"

    def begin_ready(self) -> None:
        self._nav_actions.zero_()
        self.mode = "ready"

    def ready(self) -> bool:
        """Arms and shrouds within 0.02 rad of the dig's start pose."""
        q = self.robot.data.joint_pos.torch[0]
        arms = (q[self._arm_ids] - self.cfg.arm_start_angle).abs()
        return bool((arms < 0.02).all() & (q[self._shroud_ids].abs() < 0.02).all())

    # --- where things are, env-local ---

    def facing_x(self) -> int:
        """1 if the machine's front faces +x, else -1."""
        return 1 if float(torch.cos(self._base_yaw()[0])) > 0.0 else -1

    def lane_y(self, iy: int) -> float:
        """Lane iy's centre line, env-local; lane 0 is the one at -y."""
        return (iy - 0.5 * (self.lanes - 1)) * self.cfg.work_area_width

    def anchor_x(self, ix: int, facing: int) -> float:
        """Patch ix's pass anchor for a pass toward facing: work_area_behind
        into the patch from the end the pass starts at."""
        start = self.pit_x + ix * self.patch_length
        behind = self.cfg.work_area_behind
        return start + behind if facing > 0 else start + self.patch_length - behind

    def pass_start(self, iy: int, ix: int, facing: int) -> tuple[float, float]:
        """Where the chassis stands to start lane iy's patch ix toward facing,
        the front drum over its anchor at the arm angle the pass starts from."""
        return (self.anchor_x(ix, facing) - facing * self.dig_reach, self.lane_y(iy))

    def dump_route(self, y: float) -> list[tuple[float, float]]:
        """From lane y in the pit: back out to the yard, up to the way round and
        along it to the next dump stop."""
        x = self.pit_x + PILE_STEP * (self._dumps % PILES)
        return [(self.yard_x, y), (self.yard_x, self.way), (x, self.way)]

    def pit_mask(self) -> torch.Tensor:
        """(ny, nx) the pit's grid cells."""
        x, y = self._grid_xy[..., 0], self._grid_xy[..., 1]
        return (x >= self.pit_x) & (x <= self.pit_x + self.cfg.pit_length) & (y.abs() <= self.half_width)

    def patch_level(self) -> float:
        """The last pass's work area: its mean height relative to the fresh bed, m."""
        return float((self._bed_grid[0][self._work_mask[0]] - self.cfg.bed_top).mean())

    # --- navigate ---

    def nav_obs(self) -> torch.Tensor:
        """The navigate policy's observation, its goal the route's next waypoint
        and its path the straight line there, as on navigate's flat cells.
        Within WAYPOINT_RADIUS of a waypoint, the goal moves on to the next."""
        n = self._nav
        o = self.scene.env_origins[:, :2]
        root = self.robot.data.root_pos_w.torch[:, :2]
        here = (root - o)[0]
        last = self._route.shape[0] - 1
        while self._route_next < last and float(torch.linalg.norm(self._route[self._route_next] - here)) < WAYPOINT_RADIUS:
            self._route_next += 1
        goal = self._route[self._route_next]
        ahead = along(torch.stack([here, goal]), n.path_lookahead) + o
        return navigate_actor_obs(
            self._nav_spec, self._proprio(), root, self._base_yaw(), goal[None] + o,
            self._route_yaw[self._route_next - 1].reshape(1), ahead[None], self._nav_actions,
            n.goal_dist_max, n.goal_near_range,
        )

    def _goal_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(E,) distance to the goal and heading error to its axis."""
        dist = torch.linalg.norm(self._goal_xy - self.robot.data.root_pos_w.torch[:, :2], dim=-1)
        return dist, R.axis_heading_error(self._goal_yaw, self._base_yaw())

    def goal_error(self) -> tuple[float, float]:
        """Distance to the goal and heading error to its axis, either end forward."""
        dist, herr = self._goal_terms()
        return float(dist[0]), float(herr[0])

    def arrived(self, dist_tol: float | None = None, heading_tol: float | None = None) -> bool:
        """Inside the goal tolerances, navigate's unless given, and stopped,
        arrive_hold_steps steps in a row. Call once a step."""
        n, d = self._nav, self.robot.data
        dist, herr = self._goal_terms()
        still = R.goal_reached(
            dist, herr, torch.linalg.norm(d.root_lin_vel_b.torch[:, :2], dim=-1), d.root_ang_vel_b.torch[:, 2],
            n.goal_dist_tol if dist_tol is None else dist_tol,
            n.goal_heading_tol_rad if heading_tol is None else heading_tol,
            n.goal_speed_tol, n.goal_yaw_rate_tol,
        )
        self._held = self._held + 1 if bool(still[0]) else 0
        return self._held >= n.arrive_hold_steps

    def dumped(self) -> bool:
        return bool(self._fill_kg[0, 0] < 0.5)

    # --- actions ---

    def _pre_physics_step(self, actions: torch.Tensor):
        if self.mode == "dig":
            return super()._pre_physics_step(actions)
        s = self._nav.action_smoothing
        drive = actions[:, :2].clamp(-1.0, 1.0) if self.mode == "drive" else torch.zeros_like(self._nav_actions)
        self._nav_actions[:] = (1.0 - s) * drive + s * self._nav_actions

    def _apply_action(self):
        if self.mode == "dig":
            return super()._apply_action()
        n, dev = self.num_envs, self.device
        self._apply_drive(self._nav_actions[:, 0], self._nav_actions[:, 1])
        if self.mode == "ready":
            self._apply_arms(torch.full((n,), self.cfg.arm_start_angle, device=dev))
            self._apply_shrouds(torch.zeros(n, device=dev))
            self._apply_drums(torch.zeros(n, device=dev))
            self._step_soil()
            return
        self._apply_arms(torch.full((n,), self._nav.arm_hold_angle, device=dev))
        if self.mode == "drive":
            self._apply_drums(torch.zeros(n, device=dev))
            self._apply_shrouds(torch.full((n,), math.pi, device=dev))
        else:
            self._pour()
        self._step_soil()

    def _pour(self) -> None:
        """Front inlet down and drum reversed; once the inlet is there, its
        load goes onto the bed under it at dump_rate."""
        c, d, dev = self.cfg, self.robot.data, self.device
        pour = c.dump_tilt - self._nav.arm_hold_angle
        self._apply_shrouds(torch.tensor([[pour, math.pi]], device=dev))
        self._apply_drums(torch.tensor([[-c.dig_drum_speed, 0.0]], device=dev))
        if abs(float(d.joint_pos.torch[0, self._shroud_ids[0]]) - pour) > 0.1:
            return
        kg = self._fill_kg[:, 0].clamp(max=c.dump_rate * self._dig_dt)
        body = self._shroud_body_ids[0]
        centre = d.body_link_pos_w.torch[:, body] - self.scene.env_origins
        axis = rft.drum_axis(d.body_link_quat_w.torch[:, body])
        self._bed_grid, self._loose = rft.deposit(self._bed_grid, self._loose, self._cells, centre, axis, kg, self._rft)
        self._fill_kg[:, 0] -= kg

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Never done: the runner ends the run. The pass's end and failures are still read."""
        super()._get_dones()
        none = torch.zeros_like(self._pass_done)
        return none, none


def run_site(env, dig, nav, site, record=None, max_minutes: float = 60.0) -> tuple[list[dict], str]:
    """Dig a planner's site. Before each pass the machine is lined up along a
    lane; the planner picks a cut it can make facing the way it faces (it turns
    on the spot if there is none) and the level to cut to. The machine drives
    in along the cut's lane to the pass start, sets its tools to the dig's
    start pose and digs the pass; with soil in the drum it backs out to the
    yard, goes round to a pile and empties it. Each patch's height is read back
    from the bed. dig and nav map their observations to actions; record(env)
    is called after every step. Returns a row per trim and how the run ended."""
    u = env.unwrapped
    if abs(site.cfg.patch_x - u.patch_length) > 1e-6:
        raise ValueError(f"the planner's patch is {site.cfg.patch_x} m, the dig's work area {u.patch_length} m")
    if len(site.height) != u.lanes:
        raise ValueError(f"the planner's site has {len(site.height)} lanes, the bed {u.lanes}")
    dt, voxel = u.step_dt, site.cfg.voxel
    zero = torch.zeros(1, u.cfg.action_space, device=u.device)
    state = {"obs": env.reset()[0], "steps": 0}
    limit = int(max_minutes * 60.0 / dt)

    def now() -> float:
        return state["steps"] * dt

    def step(action: torch.Tensor) -> None:
        state["obs"] = env.step(action)[0]
        state["steps"] += 1
        if record is not None:
            record(u)
        if bool(u._failed_now[0] | u._nonfinite_now[0]):
            raise RuntimeError(f"the machine failed at t = {now():.1f} s: tipped, off the bed or non-finite")
        if state["steps"] >= limit:
            raise RuntimeError(f"out of time after {max_minutes:g} min")

    def here() -> tuple[float, float]:
        xy = u.robot.data.root_pos_w.torch[0, :2] - u.scene.env_origins[0, :2]
        return float(xy[0]), float(xy[1])

    def drive(route, yaw: float, loose: bool = False) -> dict:
        """To the route's end; a loose stop is done within LOOSE_M and LOOSE_DEG."""
        u.begin_drive(route, yaw)
        tol = (LOOSE_M, math.radians(LOOSE_DEG)) if loose else (None, None)
        t0 = now()
        for _ in range(int(DRIVE_SECONDS / dt)):
            a = zero.clone()
            a[:, :2] = nav(u.nav_obs()).clamp(-1.0, 1.0)
            step(a)
            if u.arrived(*tol):
                break
        dist, herr = u.goal_error()
        goal = tuple(round(v, 2) for v in route[-1])
        if dist > LOOSE_M or abs(herr) > math.radians(LOOSE_DEG):
            raise RuntimeError(f"did not reach {goal} in {DRIVE_SECONDS:g} s: {dist:.2f} m, {math.degrees(herr):+.1f} deg off")
        return {"s": round(now() - t0, 1), "miss_m": round(dist, 3), "miss_deg": round(math.degrees(herr), 1)}

    def to_yard(y: float) -> dict:
        """Lined up along x on lane line y at the yard: from the way round down
        to it, or out of the pit along the lane it is on and across."""
        x, at = here()
        if at > u.way - 0.5:
            route = [(u.yard_x, u.way), (u.yard_x, y)]
        elif abs(at - y) > 0.3:
            route = [(u.yard_x, at), (u.yard_x, y)]
        else:
            route = [(u.yard_x, y)]
        return drive(route, 0.0, loose=True)

    def turn_round() -> None:
        """A quarter turn on the spot and back: navigate takes either end of a
        heading, so half the time the machine ends up facing the other way."""
        spot = [here()]
        drive(spot, 0.5 * math.pi, loose=True)
        drive(spot, 0.0, loose=True)

    def dig_pass(iy: int, ix: int, level: float, facing: int) -> dict:
        u.begin_ready()
        for _ in range(int(READY_SECONDS / dt)):
            step(zero)
            if u.ready():
                break
        depth = u.begin_pass(u.anchor_x(ix, facing), u.lane_y(iy), u.cfg.bed_top + level * voxel, facing)
        t0 = now()
        for _ in range(int(PASS_SECONDS / dt)):
            step(dig(state["obs"]["policy"]).clamp(-1.0, 1.0))
            if bool(u._pass_done[0]):
                break
        return {
            "toward": "+x" if facing > 0 else "-x", "depth_cm": round(100 * depth, 1), "pass_s": round(now() - t0, 1),
            "removed_kg": round(float(u._removed_kg[0]), 1), "held_kg": round(float(u._payload_kg[0]), 1),
            "shape_done": bool(u._shape_done[0]), "floor_cm": round(100 * u.patch_level(), 1),
        }

    def dump() -> float:
        before = float(u._fill_kg[0, 0])
        u.begin_dump()
        for _ in range(int(DUMP_SECONDS / dt)):
            step(zero)
            if u.dumped():
                break
        return round(before - float(u._fill_kg[0, 0]), 1)

    rows: list[dict] = []
    planned = sum(h - t for hr, tr in zip(site.height, site.target) for h, t in zip(hr, tr))
    lane = u.lanes // 2                 # the lane line the machine is on
    at_pile, turns, detours = False, 0, 0
    try:
        while site.pending() and site.trims < 3 * planned:
            if at_pile:
                to_yard(u.lane_y(lane))
                at_pile = False
            facing = u.facing_x()
            pick = site.next_trim(facing)
            if pick is None:
                if site.next_trim(-facing) is None or turns >= TURN_TRIES:
                    break
                if here()[0] > u.yard_x + 0.5:
                    to_yard(u.lane_y(lane))
                turn_round()
                turns += 1
                continue
            turns = 0
            (iy, ix), level = pick
            if iy != lane:
                to_yard(u.lane_y(iy))
                lane = iy
                if u.facing_x() != facing:          # turned round crossing over: choose again
                    continue
            start = u.pass_start(iy, ix, facing)
            row = {"trim": site.trims + 1, "lane": iy, "patch": ix, "target_cm": round(100 * level * voxel, 1)}
            if math.dist(here(), start) > 0.05:
                row["to_pass"] = drive([start], 0.0)
            if u.facing_x() != facing:              # turned round on the way in: choose again
                detours += 1
                if detours > 2 * TURN_TRIES:
                    raise RuntimeError("kept turning round on the way into the pit")
                continue
            detours = 0
            row.update(dig_pass(iy, ix, level, facing))
            site.trim((iy, ix), height=row["floor_cm"] / 100 / voxel)
            rows.append(row)
            print(f"[site] trim {row['trim']:3d}, lane {iy} patch {ix} toward {row['toward']}: to {row['target_cm']:+.1f} cm, "
                  f"cut {row['depth_cm']:.1f} cm in {row['pass_s']:.0f} s -> floor {row['floor_cm']:+.1f} cm; "
                  f"{row['removed_kg']:.0f} kg removed, {row['held_kg']:.0f} held; t = {now():.0f} s", flush=True)
            if float(u._fill_kg[0, 0]) >= DUMP_MIN_KG:
                row["to_dump"] = drive(u.dump_route(u.lane_y(iy)), 0.5 * math.pi, loose=True)
                row["dumped_kg"] = dump()
                at_pile = True
                print(f"[site]           dumped {row['dumped_kg']:.0f} kg; t = {now():.0f} s", flush=True)
    except RuntimeError as err:
        return rows, f"stopped: {err}"
    if site.done():
        return rows, "done"
    aside = [p for p in site.set_aside()]
    why = (f"; patches {aside} set aside after {site.cfg.retries} failed trims in a row" if aside
           else f"; could not turn round in {TURN_TRIES} tries" if turns >= TURN_TRIES
           else "; nothing left it can cut either way round")
    return rows, f"stopped after {site.trims} trims with {len(site.pending())} patches short{why}"
