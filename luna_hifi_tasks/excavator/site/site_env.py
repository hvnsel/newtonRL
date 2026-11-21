# site_env.py
#
# One machine on a large bed of the RFT soil, never reset, digging a trench
# with the trained policies. The runner puts the env in one of three modes:
#
#   dig    a pass of the excavate policy. begin_pass() puts the work area and
#          the plane where the planner says; the settle, ramp, line hold and
#          actions are the excavate env's, as in training.
#   drive  the navigate policy, along a route to a goal pose. nav_obs() is its
#          observation; its [forward, yaw] drive the wheels as in navigate, the
#          arms stowed, the drums still, the inlets up.
#   dump   scripted: wheels held, the front inlet turned down and its drum
#          reversed, its load laid on the bed under it.
#
# The soil steps in every mode, with the wheels on it.
#
# The trench runs along +x from trench_x, centred on y = 0, and the machine
# straddles it, its wheels on the ground either side. It spawns at the origin
# facing +x, at the start of the first patch's pass. Past the far end it turns
# left to dump on one of PILES piles beside the way back, which runs LANE to
# the left of the trench, turns back into its line and stops on it, lined up,
# before it drives in to the next pass.
#
# run_site() digs a planner's site (luna_hifi_tasks/planner/site.py), reading
# each patch's height back from the bed after its pass.

from __future__ import annotations

import math

import torch

from isaaclab.utils.configclass import configclass

from ..excavate.excavate_env_cfg import BED_FLOOR_Z, ExcavateRftSceneCfg, ExcavatorExcavateRftEnvCfg
from ..excavate.excavate_rft_env import ExcavatorExcavateRftEnv
from ..excavator import ARM_LEN, PIVOT_X
from ..excavator_cfg import ARM_STOW_ANGLE
from ..mdp import rewards as R
from ..mdp import rft
from ..mdp.observations import navigate_actor_obs, navigate_obs_spec
from ..navigate.navigate_env_cfg import ExcavatorNavigateEnvCfg

REACH = PIVOT_X + ARM_LEN * math.cos(ARM_STOW_ANGLE)   # m the front drum sits ahead of the chassis, stowed
LANE = 1.5                  # m left of the trench: the way back
EXIT = 2.0                  # m past the trench end: where the machine turns to dump, its rear wheels clear
STAGE = 0.6                 # m short of the trench the front drum stops, lined up, before driving in
TURN = 1.5                  # m behind that stop the way back turns into the trench line
PILES, PILE_STEP = 3, 0.6   # the dumps go round three piles, this far apart along x
WAYPOINT_RADIUS = 0.5       # m: a route waypoint counts as passed this close
DRIVE_SECONDS, PASS_SECONDS, DUMP_SECONDS = 90.0, 40.0, 15.0


@configclass
class ExcavatorSiteEnvCfg(ExcavatorExcavateRftEnvCfg):
    """One machine on a bed laid out round a trench_length trench, its piles
    and the way round."""

    scene: ExcavateRftSceneCfg = ExcavateRftSceneCfg(num_envs=1, env_spacing=5.12, replicate_physics=True)
    episode_length_s = 1.0e5
    trench_length: float = 2.4                 # m
    spawn_y_jitter = 0.0
    soil_reset_every: int = 1_000_000
    bed_margin_x: float = 0.5
    bed_margin_y: float = 0.5
    max_env_excursion = 12.0
    # Dumping: the front inlet dump_tilt off straight down, its load laid out at dump_rate.
    dump_tilt: float = 0.0                     # rad
    dump_rate: float = 30.0                    # kg/s

    def __post_init__(self) -> None:
        x0 = REACH - self.work_area_behind
        self.bed_x = (x0 - STAGE - REACH - TURN - 1.8, x0 + self.trench_length + EXIT + PILE_STEP * (PILES - 1) + 1.8)
        self.bed_y = (-1.8, LANE + REACH + 1.0)
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
        self.trench_x = REACH - cfg.work_area_behind
        self.patch_length = cfg.work_area_behind + cfg.work_area_length
        self._nav = ExcavatorNavigateEnvCfg()
        self._nav_spec = navigate_obs_spec()
        self._nav_actions = torch.zeros(1, 2, device=dev)
        self._route = torch.zeros(1, 2, device=dev)
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

    def begin_pass(self, anchor_x: float, plane_z: float) -> float:
        """A dig pass along +x, its work area anchored at (anchor_x, 0) and its
        plane at plane_z, env-local: the depth below the work area's mean
        surface, kept inside the trained range. Returns that depth."""
        ids = torch.zeros(1, dtype=torch.long, device=self.device)
        o = self.scene.env_origins
        anchor = o[:, :2] + torch.tensor([[anchor_x, 0.0]], device=self.device)
        yaw = torch.zeros(1, device=self.device)
        self._set_work_area(ids, anchor, yaw)
        work = self._work_mask & (self._bed_grid > BED_FLOOR_Z + 0.5 * self.cfg.voxel_size)
        mean = (self._bed_grid * work).sum(dim=(1, 2)) / work.sum(dim=(1, 2)).clamp_min(1)
        lo, hi = self.cfg.cut_depth_range
        depth = (mean - plane_z).clamp(lo, hi)
        arm = self.robot.data.joint_pos.torch[:, self._arm_ids[0]]
        self._begin_pass(ids, anchor, o[:, 2] + mean, yaw, arm, depth, torch.zeros(1, 2, device=self.device))
        self.mode = "dig"
        return float(depth[0])

    def begin_drive(self, route: list[tuple[float, float]], goal_yaw: float) -> None:
        """Drive along env-local waypoints to a goal pose at the last, facing
        along goal_yaw (either way along it, as navigate trains)."""
        self._route = torch.tensor(route, device=self.device)
        self._route_next = 0
        self._goal_xy = self.scene.env_origins[:, :2] + self._route[-1]
        self._goal_yaw[:] = goal_yaw
        self._nav_actions.zero_()
        self._held = 0
        self.mode = "drive"

    def begin_dump(self) -> None:
        self._dumps += 1
        self.mode = "dump"

    # --- where things are, env-local ---

    def anchor_x(self, ix: int) -> float:
        """Patch ix's pass anchor: work_area_behind into the patch."""
        return self.trench_x + ix * self.patch_length + self.cfg.work_area_behind

    def pass_start(self, ix: int) -> tuple[float, float]:
        """Where the chassis stands to start patch ix's pass, the front drum over its anchor."""
        return (self.anchor_x(ix) - REACH, 0.0)

    def dump_route(self) -> list[tuple[float, float]]:
        """On past the trench end, then left to face the next pile."""
        x = self.trench_x + self.cfg.trench_length + EXIT + PILE_STEP * (self._dumps % PILES)
        return [(x, 0.0), (x, LANE)]

    def staging(self) -> tuple[float, float]:
        """Where the chassis stops on the trench line before driving in, the front drum STAGE short of the trench."""
        return (self.trench_x - STAGE - REACH, 0.0)

    def return_route(self) -> list[tuple[float, float]]:
        """Back along the lane, round behind the trench and up its line to the staging stop."""
        x = self.staging()[0] - TURN
        return [(x, LANE), (x, 0.0), self.staging()]

    def trench_mask(self) -> torch.Tensor:
        """(ny, nx) the trench's grid cells."""
        x, y = self._grid_xy[..., 0], self._grid_xy[..., 1]
        return (x >= self.trench_x) & (x <= self.trench_x + self.cfg.trench_length) & (y.abs() <= 0.5 * self.cfg.work_area_width)

    def patch_level(self) -> float:
        """The last pass's work area: its mean height relative to the fresh bed, m."""
        return float((self._bed_grid[0][self._work_mask[0]] - self.cfg.bed_top).mean())

    # --- navigate ---

    def nav_obs(self) -> torch.Tensor:
        """The navigate policy's observation: the path the points path_lookahead
        along the route from the chassis, and the goal pose."""
        n = self._nav
        o = self.scene.env_origins[:, :2]
        root = self.robot.data.root_pos_w.torch[:, :2]
        here = (root - o)[0]
        last = self._route.shape[0] - 1
        while self._route_next < last and float(torch.linalg.norm(self._route[self._route_next] - here)) < WAYPOINT_RADIUS:
            self._route_next += 1
        ahead = along(torch.cat([here[None], self._route[self._route_next:]]), n.path_lookahead) + o
        return navigate_actor_obs(
            self._nav_spec, self._proprio(), root, self._base_yaw(), self._goal_xy, self._goal_yaw,
            ahead[None], self._nav_actions, n.goal_dist_max, n.goal_near_range,
        )

    def _goal_terms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(E,) distance to the goal, heading error to its axis, and the yaw."""
        yaw = self._base_yaw()
        dist = torch.linalg.norm(self._goal_xy - self.robot.data.root_pos_w.torch[:, :2], dim=-1)
        return dist, R.axis_heading_error(self._goal_yaw, yaw), yaw

    def goal_error(self) -> tuple[float, float, bool]:
        """Distance to the goal, heading error to its axis, and whether the
        machine faces along goal_yaw rather than against it."""
        dist, herr, yaw = self._goal_terms()
        return float(dist[0]), float(herr[0]), bool(torch.cos(yaw - self._goal_yaw)[0] > 0.0)

    def arrived(self) -> bool:
        """Inside navigate's goal tolerances and stopped, arrive_hold_steps
        steps in a row. Call once a step."""
        n, d = self._nav, self.robot.data
        dist, herr, _ = self._goal_terms()
        still = R.goal_reached(
            dist, herr, torch.linalg.norm(d.root_lin_vel_b.torch[:, :2], dim=-1), d.root_ang_vel_b.torch[:, 2],
            n.goal_dist_tol, n.goal_heading_tol_rad, n.goal_speed_tol, n.goal_yaw_rate_tol,
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
    """Dig a planner's site: for each trim it picks, drive round to the
    staging stop and along the trench line to the patch's pass start, dig the
    pass, drive past the trench to a pile and empty the drum;
    each patch's height is read back from the bed. dig and nav map their
    observations to actions; record(env) is called after every step. Returns
    a row per trim and how the run ended."""
    u = env.unwrapped
    if abs(site.cfg.patch_x - u.patch_length) > 1e-6:
        raise ValueError(f"the planner's patch is {site.cfg.patch_x} m, the dig's work area {u.patch_length} m")
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

    def drive(route, yaw: float) -> dict:
        u.begin_drive(route, yaw)
        t0 = now()
        for _ in range(int(DRIVE_SECONDS / dt)):
            a = zero.clone()
            a[:, :2] = nav(u.nav_obs()).clamp(-1.0, 1.0)
            step(a)
            if u.arrived():
                break
        dist, herr, forward = u.goal_error()
        goal = tuple(round(v, 2) for v in route[-1])
        if not forward:
            raise RuntimeError(f"reached {goal} facing the wrong way")
        if dist > 0.25 or abs(herr) > math.radians(10.0):
            raise RuntimeError(f"did not reach {goal} in {DRIVE_SECONDS:g} s: {dist:.2f} m, {math.degrees(herr):+.1f} deg off")
        return {"s": round(now() - t0, 1), "miss_m": round(dist, 3), "miss_deg": round(math.degrees(herr), 1)}

    def dig_pass(ix: int, height: int) -> dict:
        depth = u.begin_pass(u.anchor_x(ix), u.cfg.bed_top + height * voxel)
        t0 = now()
        for _ in range(int(PASS_SECONDS / dt)):
            step(dig(state["obs"]["policy"]).clamp(-1.0, 1.0))
            if bool(u._pass_done[0]):
                break
        return {
            "depth_cm": round(100 * depth, 1), "pass_s": round(now() - t0, 1),
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
    first = True
    try:
        while (options := site.actionable()) and site.trims < 3 * planned:
            iy, ix = min(options, key=site.cycle_seconds)
            h = site.height[iy][ix]
            row = {"trim": site.trims + 1, "patch": ix, "target_cm": round(100 * (h - 1) * voxel, 1)}
            if not first:
                row["to_line"] = drive(u.return_route(), 0.0)
            if not (first and ix == 0):
                row["to_pass"] = drive([u.pass_start(ix)], 0.0)
            row.update(dig_pass(ix, h - 1))
            site.trim((iy, ix), height=round(row["floor_cm"] / 100 / voxel))
            rows.append(row)
            print(f"[site] trim {row['trim']:2d}, patch {ix}: to {row['target_cm']:+.0f} cm, cut {row['depth_cm']:.1f} cm "
                  f"in {row['pass_s']:.0f} s -> floor {row['floor_cm']:+.1f} cm; {row['removed_kg']:.0f} kg removed, "
                  f"{row['held_kg']:.0f} held; t = {now():.0f} s", flush=True)
            row["to_dump"] = drive(u.dump_route(), 0.5 * math.pi)
            row["dumped_kg"] = dump()
            first = False
            print(f"[site]          dumped {row['dumped_kg']:.0f} kg; t = {now():.0f} s", flush=True)
    except RuntimeError as err:
        return rows, f"stopped: {err}"
    return rows, "done" if site.done() else f"stopped after {site.trims} trims with {len(site.pending())} patches short"
