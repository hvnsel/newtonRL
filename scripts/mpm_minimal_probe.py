#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Minimal, Isaac-Lab-free reproduction + discriminating instrumentation for the
Newton 1.6.1 SolverImplicitMPM sinkage/creep bug.

NO Isaac Lab. NO articulation. NO SolverCoupledProxy. NO MuJoCo.
Just SolverImplicitMPM + a static box container + (optionally) ONE rigid box
whose 1-DOF vertical motion is integrated here, on the host, in plain numpy,
so that every term in the body's force balance is visible.

Reference material this was written against (newton 1.6.1 source):
  newton/examples/mpm/example_mpm_twoway_coupling.py      (coupling pattern)
  newton/tests/test_implicit_mpm.py                       (driving in isolation)
  newton/_src/solvers/implicit_mpm/solve_rheology.py      (solver loop)
  newton/_src/solvers/implicit_mpm/contact_solver_kernels.py (contact law)
  newton/_src/solvers/implicit_mpm/rasterized_collisions.py  (sdf, project_outside)

--------------------------------------------------------------------------
TESTS
--------------------------------------------------------------------------
  env         Report device / graph capability / which solver loop will run and
              whether max_iterations // solve_granularity == 0  (see MECHANISM C).
  geostatic   Bed under gravity, NO rigid body. Does it hold itself up, and is
              sigma_zz linear in depth with slope rho*g?
  dt          One box, free sinking, dt swept. Rate-independent plasticity =>
              same equilibrium sinkage. Separates (a) from (b)/(c).
  hold        One KINEMATIC box held at a fixed overlap. Position-level contact
              => reaction grows with overlap. Velocity-level => it does not.
              Separates (b) from (a).
  iters       max_iterations / tolerance sweep. Sinkage falls => (c).
  basis       strain_basis x velocity_basis sweep (LBB / locking check).
  project     Free sinking with vs without solver.project_outside() -- the
              position-level stabilizer that every other Newton MPM example
              calls and the two-way-coupling example does not.
  all         env, geostatic, dt, hold, iters, basis, project

--------------------------------------------------------------------------
EXACT COMMAND LINES  (RTX 3090, project venv)
--------------------------------------------------------------------------
  PY=~/IsaacLab/.venv/bin/python
  S=/home/user/newtonRL/scripts/mpm_minimal_probe.py

  $PY $S env
  $PY $S geostatic --seconds 2.0
  $PY $S geostatic --seconds 2.0 --host-loop --max-iterations 250
  $PY $S dt        --seconds 4.0
  $PY $S hold
  $PY $S iters     --seconds 4.0
  $PY $S basis     --seconds 4.0
  $PY $S project   --seconds 4.0
  $PY $S all       --seconds 4.0 --json /tmp/mpm_probe.json

Add --host-loop to any test to force the host (non-CUDA-graph) solver loop,
which is the ONLY path that reports a per-batch residual. READ THE WARNING
THAT env PRINTS BEFORE TRUSTING --host-loop WITH max_iterations < 25.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM

# ---------------------------------------------------------------------------
# Instrumentation: exposes the residual and the iteration count that the
# public API does not report. Zero edits to the newton source tree.
# ---------------------------------------------------------------------------

import newton._src.solvers.implicit_mpm.solve_rheology as _sr


@dataclass
class SolveTrace:
    """Per-newton-step record of what the rheology solver actually did."""

    residual_calls: list[tuple[float, float]] = field(default_factory=list)
    rheology_solve_calls: int = 0
    contact_solve_calls: int = 0
    granularity: int | None = None
    max_iterations: int | None = None
    predicted_host_batches: int | None = None

    def reset(self):
        self.residual_calls.clear()
        self.rheology_solve_calls = 0
        self.contact_solve_calls = 0


TRACE = SolveTrace()


def install_instrumentation():
    """Patch three private seams in solve_rheology to make the solve observable.

    1. _nonlinear_solver_result_norms  -- called once per residual evaluation in
       BOTH the graph and the host loop (only when verbose=True). Gives
       (res_l2, res_linf).
    2. _run_solver_loop -- records max_iterations and the solver's declared
       solve_granularity, and hence whether the host loop will run ZERO batches.
    3. _RheologySolver.solve / _ContactSolver.solve -- counts actual iterations.
       NOTE: in CUDA-graph mode these are called only while the graph is being
       recorded, so the count reflects the recorded body, not the replays.
       Use --host-loop for a trustworthy count.
    """
    if getattr(_sr, "_probe_instrumented", False):
        return

    orig_norms = _sr._nonlinear_solver_result_norms

    def norms(residual, scale):
        out = orig_norms(residual, scale)
        try:
            TRACE.residual_calls.append((float(out[0]), float(out[1])))
        except Exception:
            pass
        return out

    _sr._nonlinear_solver_result_norms = norms

    orig_loop = _sr._run_solver_loop

    def loop(rheology_solver, contact_solver, max_iterations, tolerance, *a, **kw):
        try:
            g = int(rheology_solver.solve_granularity)
            TRACE.granularity = g
            TRACE.max_iterations = int(max_iterations)
            TRACE.predicted_host_batches = int(max_iterations) // g
        except Exception:
            pass
        return orig_loop(rheology_solver, contact_solver, max_iterations, tolerance, *a, **kw)

    _sr._run_solver_loop = loop

    for cls, counter in ((_sr._RheologySolver, "rheology_solve_calls"), (_sr._ContactSolver, "contact_solve_calls")):
        for sub in (cls, *_all_subclasses(cls)):
            fn = sub.__dict__.get("solve")
            if fn is None:
                continue

            def make(fn=fn, counter=counter):
                def wrapped(self, *a, **kw):
                    setattr(TRACE, counter, getattr(TRACE, counter) + 1)
                    return fn(self, *a, **kw)

                return wrapped

            sub.solve = make()

    _sr._probe_instrumented = True


def _all_subclasses(cls):
    out = []
    for s in cls.__subclasses__():
        out.append(s)
        out.extend(_all_subclasses(s))
    return out


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------

G_LUNAR = 1.62


@dataclass
class Cfg:
    # soil (matches excavate_env_cfg.py as spawned)
    density: float = 6000.0
    friction: float = 0.84
    yield_stress: float = 500.0
    yield_pressure: float = 1.0e12
    young_modulus: float = 1.0e8
    poisson: float = 0.3
    viscosity: float = 0.0
    dilatancy: float = 0.0
    damping: float = 0.0

    # bed  (SMALL: 0.60 x 0.60 x 0.24 m, ~1350 particles at ppc 8)
    bed_x: float = 0.60
    bed_y: float = 0.60
    bed_z: float = 0.24
    particles_per_cell: int = 8
    wall_mu: float = 0.0  # frictionless walls => sigma_zz == rho g z exactly
    floor_mu: float = 1.0

    # solver (matches MPMSolverCfg)
    voxel_size: float = 0.08
    grid_type: str = "sparse"
    grid_padding: int = 0
    critical_fraction: float = 0.90
    strain_basis: str = "P0"
    velocity_basis: str = "Q1"
    collider_basis: str = "pic27"
    transfer_scheme: str = "apic"
    collider_velocity_mode: str = "forward"
    max_iterations: int = 24
    tolerance: float = 1.0e-4

    # rigid box
    box_hx: float = 0.10
    box_hy: float = 0.10
    box_hz: float = 0.05
    target_bearing_kpa: float = 10.0  # == rover wheel: 797/4 N over 0.2x0.1 m
    box_mass: float | None = None  # derived from target_bearing_kpa if None

    # driving
    dt: float = 0.01
    seconds: float = 4.0
    gravity: float = G_LUNAR

    # knobs
    host_loop: bool = False
    verbose_solver: bool = False
    project_outside: bool = False
    subtract_body_force: bool = True

    def derived_box_mass(self) -> float:
        if self.box_mass is not None:
            return self.box_mass
        area = (2.0 * self.box_hx) * (2.0 * self.box_hy)
        return self.target_bearing_kpa * 1.0e3 * area / self.gravity

    def bearing_kpa(self) -> float:
        area = (2.0 * self.box_hx) * (2.0 * self.box_hy)
        return self.derived_box_mass() * self.gravity / area / 1.0e3


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_soil_builder(cfg: Cfg):
    """A ModelBuilder holding only MPM particles, z-up, lunar gravity."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -cfg.gravity))
    SolverImplicitMPM.register_custom_attributes(b)

    # particle spacing from particles-per-cell (ppc = (voxel/spacing)^3)
    ppc_lin = cfg.particles_per_cell ** (1.0 / 3.0)
    spacing = cfg.voxel_size / ppc_lin
    nx = max(1, int(round(cfg.bed_x / spacing)))
    ny = max(1, int(round(cfg.bed_y / spacing)))
    nz = max(1, int(round(cfg.bed_z / spacing)))
    cx, cy, cz = cfg.bed_x / nx, cfg.bed_y / ny, cfg.bed_z / nz

    # implicit_mpm_model._compute_particle_volume_density: volume = 8 r^3,
    # so radius must be half the spacing for volume == cell volume.
    radius = 0.5 * max(cx, cy, cz)
    mass = cx * cy * cz * cfg.density

    b.add_particle_grid(
        pos=wp.vec3(-0.5 * cfg.bed_x + 0.5 * cx, -0.5 * cfg.bed_y + 0.5 * cy, 0.5 * cz),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=nx,
        dim_y=ny,
        dim_z=nz,
        cell_x=cx,
        cell_y=cy,
        cell_z=cz,
        mass=mass,
        jitter=0.0,
        radius_mean=radius,
        custom_attributes={
            "mpm:young_modulus": cfg.young_modulus,
            "mpm:poisson_ratio": cfg.poisson,
            "mpm:damping": cfg.damping,
            "mpm:friction": cfg.friction,
            "mpm:yield_stress": cfg.yield_stress,
            "mpm:yield_pressure": cfg.yield_pressure,
            "mpm:tensile_yield_ratio": 0.0,
            "mpm:hardening": 0.0,
            "mpm:dilatancy": cfg.dilatancy,
            "mpm:viscosity": cfg.viscosity,
        },
    )
    meta = dict(n=nx * ny * nz, spacing=(cx, cy, cz), radius=radius, mass=mass, dims=(nx, ny, nz))
    return b, meta


def build_rig_builder(cfg: Cfg, with_box: bool, box_z: float, kinematic_box: bool):
    """Container (5 static boxes on body -1) + optional one rigid loading box."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -cfg.gravity))

    t = 0.10  # container wall thickness
    hx, hy, hz = 0.5 * cfg.bed_x, 0.5 * cfg.bed_y, 0.5 * cfg.bed_z

    # floor
    b.default_shape_cfg.mu = cfg.floor_mu
    b.add_shape_box(-1, xform=wp.transform(p=wp.vec3(0.0, 0.0, -t)), hx=hx + t, hy=hy + t, hz=t)
    # walls (frictionless by default so the geostatic profile is not arched)
    b.default_shape_cfg.mu = cfg.wall_mu
    for sx, sy, ex, ey in (
        (1, 0, t, hy + t),
        (-1, 0, t, hy + t),
        (0, 1, hx + t, t),
        (0, -1, hx + t, t),
    ):
        b.add_shape_box(
            -1,
            xform=wp.transform(p=wp.vec3(sx * (hx + t), sy * (hy + t), cfg.bed_z)),
            hx=ex,
            hy=ey,
            hz=cfg.bed_z + t,
        )

    body = -1
    if with_box:
        b.default_shape_cfg.mu = cfg.friction
        body = b.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, box_z)),
            mass=cfg.derived_box_mass(),
            is_kinematic=kinematic_box,
        )
        b.add_shape_box(body, hx=cfg.box_hx, hy=cfg.box_hy, hz=cfg.box_hz)
    return b, body


def make_mpm_config(cfg: Cfg) -> SolverImplicitMPM.Config:
    o = SolverImplicitMPM.Config()
    o.voxel_size = cfg.voxel_size
    o.grid_type = cfg.grid_type
    o.grid_padding = cfg.grid_padding
    o.critical_fraction = cfg.critical_fraction
    o.strain_basis = cfg.strain_basis
    o.velocity_basis = cfg.velocity_basis
    o.collider_basis = cfg.collider_basis
    o.transfer_scheme = cfg.transfer_scheme
    o.collider_velocity_mode = cfg.collider_velocity_mode
    o.max_iterations = cfg.max_iterations
    o.tolerance = cfg.tolerance
    return o


# ---------------------------------------------------------------------------
# The rig: MPM + optional one vertically-free rigid box, integrated on host
# ---------------------------------------------------------------------------


class Rig:
    def __init__(self, cfg: Cfg, with_box: bool, kinematic_box: bool = False, box_gap: float = 0.002):
        self.cfg = cfg
        self.with_box = with_box
        self.kinematic = kinematic_box

        soil_b, self.soil_meta = build_soil_builder(cfg)
        self.box_z0 = cfg.bed_z + cfg.box_hz + box_gap
        rig_b, self.body = build_rig_builder(cfg, with_box, self.box_z0, kinematic_box)

        self.soil_model = soil_b.finalize()
        self.rig_model = rig_b.finalize()

        self.state = self.soil_model.state()
        nb = int(self.rig_model.body_count)
        dev = self.soil_model.device
        self.state.body_q = wp.zeros(nb, dtype=wp.transform, device=dev)
        self.state.body_qd = wp.zeros(nb, dtype=wp.spatial_vector, device=dev)
        self.state.body_f = wp.zeros(nb, dtype=wp.spatial_vector, device=dev)
        if nb > 0:
            self.state.body_q.assign(self.rig_model.body_q)

        self.solver = SolverImplicitMPM(self.soil_model, config=make_mpm_config(cfg), verbose=cfg.verbose_solver)
        self.solver.setup_collider(
            model=self.rig_model, body_q=self.state.body_q if nb > 0 else None
        )
        if cfg.host_loop:
            # forces the host loop in solve_rheology._run_solver_loop, the only
            # path that evaluates a residual per batch.
            self.solver._use_cuda_graph = False
        self.collider_body = self.solver.collider_body_index.numpy().copy()

        self.mass = cfg.derived_box_mass()
        self.z = float(self.box_z0)
        self.vz = 0.0
        self.f_sand_prev = 0.0
        self.t = 0.0
        self.log = []

    # -- body <-> mpm plumbing -------------------------------------------------

    def _write_body(self, z: float, vz: float):
        if self.rig_model.body_count == 0:
            return
        q = self.state.body_q.numpy().copy()
        qd = self.state.body_qd.numpy().copy()
        q[self.body][2] = z
        qd[self.body][0:3] = (0.0, 0.0, vz)  # spatial_top == linear velocity
        self.state.body_q.assign(q)
        self.state.body_qd.assign(qd)

    def harvest(self) -> tuple[float, dict]:
        """Net +z force [N] the MPM applies to the loading box, plus per-body sums."""
        imp, pos, cid = self.solver.collect_collider_impulses(self.state)
        imp = imp.numpy()
        cid = cid.numpy()
        per_body = {}
        total = 0.0
        valid = cid >= 0
        if valid.any():
            bodies = np.full(cid.shape, -1, dtype=np.int64)
            inr = valid & (cid < len(self.collider_body))
            bodies[inr] = self.collider_body[cid[inr]]
            for bid in np.unique(bodies[bodies >= 0]):
                s = imp[bodies == bid].sum(axis=0) / self.cfg.dt
                per_body[int(bid)] = s.tolist()
            # static shapes (body -1) -> the container reaction
            m = valid & (bodies < 0)
            if m.any():
                per_body[-1] = (imp[m].sum(axis=0) / self.cfg.dt).tolist()
            if self.body >= 0 and self.body in per_body:
                total = per_body[self.body][2]
        return float(total), per_body

    def max_penetration(self) -> float:
        """Deepest node penetration [m] into any collider.

        rasterized_collisions.py:488 writes the per-node sdf into
        scratch.collider_distance_field and NOTHING in the solve ever reads it.
        That dead field is exactly the position-level error we want to measure.
        """
        try:
            d = self.solver._scratchpad.collider_distance_field.dof_values.numpy()
            d = d[np.isfinite(d) & (np.abs(d) < 1.0e6)]
            return float(-d.min()) if d.size else float("nan")
        except Exception:
            return float("nan")

    # -- stepping -------------------------------------------------------------

    def step(self):
        cfg = self.cfg
        dt = cfg.dt
        g = -cfg.gravity

        if self.with_box and not self.kinematic:
            # explicit, visible, lagged coupling -- the same scheme as
            # example_mpm_twoway_coupling.py (compute_body_forces then
            # subtract_body_force), written out so every term is inspectable.
            self.vz += dt * (g + self.f_sand_prev / self.mass)
            self.z += dt * self.vz
            v_for_mpm = self.vz - (dt * self.f_sand_prev / self.mass if cfg.subtract_body_force else 0.0)
            self._write_body(self.z, v_for_mpm)
        elif self.with_box:
            self._write_body(self.z, 0.0)  # kinematic hold

        TRACE.reset()
        self.solver.step(self.state, self.state, control=None, contacts=None, dt=dt)
        if cfg.project_outside:
            self.solver.project_outside(self.state, self.state, dt)

        f, per_body = self.harvest()
        self.f_sand_prev = f
        self.t += dt
        return f, per_body

    def run(self, seconds: float, sample_every: float = 0.1):
        n = int(round(seconds / self.cfg.dt))
        every = max(1, int(round(sample_every / self.cfg.dt)))
        for i in range(n):
            f, per_body = self.step()
            if i % every == 0 or i == n - 1:
                self.log.append(
                    dict(
                        t=self.t,
                        z=self.z,
                        vz=self.vz,
                        f_sand=f,
                        weight=self.mass * self.cfg.gravity if self.with_box else 0.0,
                        pen=self.max_penetration(),
                        res=TRACE.residual_calls[-1] if TRACE.residual_calls else None,
                        rheo_calls=TRACE.rheology_solve_calls,
                        surface=self.surface_z(),
                        per_body=per_body,
                    )
                )
        return self.log

    # -- measurement ----------------------------------------------------------

    def surface_z(self, quantile: float = 0.99) -> float:
        q = self.state.particle_q.numpy()
        return float(np.quantile(q[:, 2], quantile))

    def stress_profile(self, nbins: int = 6):
        """Per-depth-bin stress, from BOTH sources, vs the geostatic prediction.

        source 'solver' : state.mpm.particle_stress -- the solver's own Cauchy
                          stress in Pa (register_custom_attributes documents it
                          as such; solver_implicit_mpm.py ~3570 fills it).
        source 'F'      : the project probe's recipe -- treat
                          particle_elastic_strain as a deformation gradient and
                          apply a linear elastic law at E. Reported so the two
                          can be compared directly.
        """
        cfg = self.cfg
        q = self.state.particle_q.numpy()
        top = float(np.quantile(q[:, 2], 0.995))
        depth = top - q[:, 2]

        S = self.state.mpm.particle_stress.numpy()  # (n,3,3) Pa
        F = self.state.mpm.particle_elastic_strain.numpy()  # (n,3,3)

        E, nu = cfg.young_modulus, cfg.poisson
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu_l = E / (2.0 * (1.0 + nu))
        eps = 0.5 * (F + np.transpose(F, (0, 2, 1))) - np.eye(3)
        tr = np.trace(eps, axis1=1, axis2=2)
        Sf = 2.0 * mu_l * eps + lam * tr[:, None, None] * np.eye(3)

        edges = np.linspace(0.0, max(cfg.bed_z, depth.max()), nbins + 1)
        rows = []
        for i in range(nbins):
            m = (depth >= edges[i]) & (depth < edges[i + 1])
            if m.sum() < 5:
                continue
            d = float(depth[m].mean())
            geo = cfg.density * cfg.gravity * d

            def pack(A):
                a = A[m]
                szz = float(a[:, 2, 2].mean())
                sxx = float(0.5 * (a[:, 0, 0] + a[:, 1, 1]).mean())
                p = float(np.trace(a, axis1=1, axis2=2).mean() / 3.0)
                dev = a - (np.trace(a, axis1=1, axis2=2) / 3.0)[:, None, None] * np.eye(3)
                vm = float(np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2))).mean())
                return dict(szz=szz, sxx=sxx, p=p, vm=vm)

            rows.append(
                dict(
                    depth=d,
                    n=int(m.sum()),
                    geostatic=geo,
                    solver=pack(S),
                    Fderived=pack(Sf),
                    Fdiag_mean=float(np.mean([F[m][:, k, k].mean() for k in range(3)])),
                    Fdev_mean=float(np.abs(F[m] - np.eye(3)).mean()),
                )
            )

        # linearity of the SOLVER stress against geostatic
        if len(rows) >= 3:
            d = np.array([r["depth"] for r in rows])
            s = np.array([abs(r["solver"]["szz"]) for r in rows])
            A = np.vstack([d, np.ones_like(d)]).T
            (slope, icpt), *_ = np.linalg.lstsq(A, s, rcond=None)
            pred = A @ np.array([slope, icpt])
            ss_res = float(((s - pred) ** 2).sum())
            ss_tot = float(((s - s.mean()) ** 2).sum()) or 1.0
            fit = dict(
                slope=float(slope),
                intercept=float(icpt),
                r2=1.0 - ss_res / ss_tot,
                expected_slope=cfg.density * cfg.gravity,
                slope_ratio=float(slope) / (cfg.density * cfg.gravity),
            )
        else:
            fit = None
        return dict(rows=rows, fit=fit, top=top)

    def release(self):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_env(cfg: Cfg):
    dev = wp.get_device()
    cond = False
    try:
        cond = bool(wp.is_conditional_graph_supported())
    except Exception as e:
        cond = f"error: {e}"

    print("=" * 78)
    print("TEST env")
    print("=" * 78)
    print(f"  warp                        : {getattr(wp.config, 'version', '?')}")
    print(f"  newton                      : {getattr(newton, '__version__', '?')}")
    print(f"  device                      : {dev}  is_cuda={dev.is_cuda}")
    print(f"  is_conditional_graph_support: {cond}")

    rig = Rig(cfg, with_box=False)
    print(f"  particles                   : {rig.soil_model.particle_count} {rig.soil_meta}")
    print(f"  solver._use_cuda_graph      : {rig.solver._use_cuda_graph}")
    print(f"  solver.solver (resolved)    : {rig.solver.solver}")
    print(f"  solver.max_iterations       : {rig.solver.max_iterations}")
    print(f"  solver.tolerance            : {rig.solver.tolerance}")

    rig.solver.verbose = True
    TRACE.reset()
    rig.step()
    print(f"  TRACE.granularity           : {TRACE.granularity}")
    print(f"  TRACE.max_iterations        : {TRACE.max_iterations}")
    print(f"  max_iterations // granularity (host loop batch count) : {TRACE.predicted_host_batches}")
    print(f"  rheology solve() calls      : {TRACE.rheology_solve_calls}")
    print(f"  contact  solve() calls      : {TRACE.contact_solve_calls}")
    print(f"  residual evaluations        : {len(TRACE.residual_calls)} -> {TRACE.residual_calls}")

    if TRACE.predicted_host_batches == 0:
        print()
        print("  *** MECHANISM C SMOKING GUN ***")
        print("  solve_rheology.py:1871  `for batch in range(max_iterations // solve_granularity)`")
        print(f"  {TRACE.max_iterations} // {TRACE.granularity} == 0  =>  the HOST loop runs ZERO")
        print("  rheology and ZERO contact iterations. Only apply_initial_guess()")
        print("  (the previous step's warm-started stress and impulse, re-projected")
        print("  onto the yield surface) survives. Still true in 1.7.0.")
        print("  => if solver._use_cuda_graph is False in the real run, the MPM is")
        print("     running on warm-start persistence alone, which is exactly a")
        print("     correct total force with no correction of the configuration.")
    print()
    return dict(
        warp=getattr(wp.config, 'version', '?'),
        device=str(dev),
        conditional_graph=cond,
        use_cuda_graph=bool(rig.solver._use_cuda_graph),
        granularity=TRACE.granularity,
        max_iterations=TRACE.max_iterations,
        host_batches=TRACE.predicted_host_batches,
        rheology_calls=TRACE.rheology_solve_calls,
        residuals=TRACE.residual_calls,
        particles=int(rig.soil_model.particle_count),
    )


def test_geostatic(cfg: Cfg):
    print("=" * 78)
    print("TEST geostatic -- bed under gravity, NO rigid body")
    print("=" * 78)
    rig = Rig(cfg, with_box=False)
    rig.solver.verbose = cfg.verbose_solver
    z0 = rig.surface_z()
    q0 = rig.state.particle_q.numpy().copy()
    rig.run(cfg.seconds, sample_every=0.25)
    z1 = rig.surface_z()
    q1 = rig.state.particle_q.numpy()
    prof = rig.stress_profile()

    settle = z0 - z1
    elastic = cfg.density * cfg.gravity * cfg.bed_z * cfg.bed_z / (2.0 * cfg.young_modulus)
    ke = float(0.5 * (rig.state.particle_qd.numpy() ** 2).sum(axis=1).mean())

    print(f"  surface z      : {z0:.6f} -> {z1:.6f}   settlement {settle * 1e3:.3f} mm")
    print(f"  1-D elastic settlement for this bed at E={cfg.young_modulus:.1e}: {elastic * 1e3:.4f} mm")
    print(f"  mean |dz| particle : {float(np.abs(q1[:, 2] - q0[:, 2]).mean()) * 1e3:.3f} mm")
    print(f"  mean particle KE   : {ke:.3e} J/kg")
    print(f"  last residual      : {rig.log[-1]['res']}")
    print()
    print("  depth[m]   n      sigma_zz(solver)   sigma_zz(F-derived)   geostatic   p(solver)")
    for r in prof["rows"]:
        print(
            f"  {r['depth']:7.4f} {r['n']:6d}   {r['solver']['szz']:+16.4e}   "
            f"{r['Fderived']['szz']:+19.4e}   {r['geostatic']:9.2f}   {r['solver']['p']:+.4e}"
        )
    if prof["fit"]:
        f = prof["fit"]
        print()
        print(f"  |sigma_zz(solver)| vs depth : slope {f['slope']:.4e} Pa/m, intercept {f['intercept']:.4e} Pa")
        print(f"  expected slope rho*g        : {f['expected_slope']:.4e} Pa/m")
        print(f"  slope ratio (1.0 == correct): {f['slope_ratio']:.4f}       R^2 {f['r2']:.4f}")
        print()
        print("  VERDICT: a bed that holds itself up has slope_ratio ~ 1, R^2 ~ 1 and")
        print("           intercept ~ 0. slope_ratio ~ 0 with a large intercept means the")
        print("           stress is FLAT with depth and the geostatic balance is not being")
        print("           solved at all -- nothing downstream of this can be right.")
    print()
    return dict(z0=z0, z1=z1, settlement_mm=settle * 1e3, elastic_mm=elastic * 1e3, profile=prof, log=rig.log)


def test_dt(cfg: Cfg, dts=(0.02, 0.01, 0.005, 0.0025)):
    print("=" * 78)
    print("TEST dt -- rate independence. Separates (a) viscoplastic creep from (b)/(c).")
    print("=" * 78)
    print("  PREDICTIONS")
    print("    (a) viscoplastic / rate-dependent : sinkage(T) differs with dt only if the")
    print("        flow rule itself carries dt; with viscosity=0 it should NOT, so a dt")
    print("        dependence here falsifies 'pure material rate dependence'.")
    print("    (b) position-level drift          : drift per step ~ dt*v, steps ~ T/dt,")
    print("        so sinkage(T) is dt-INDEPENDENT.")
    print("    (c) per-step residual leak        : leak per step is fixed, steps ~ T/dt,")
    print("        so sinkage(T) scales ~ 1/dt. HALVING dt DOUBLES the sinkage.")
    out = []
    for dt in dts:
        c = Cfg(**{**cfg.__dict__, "dt": dt})
        rig = Rig(c, with_box=True)
        rig.run(c.seconds, sample_every=0.5)
        tail = [r for r in rig.log if r["t"] > 0.75 * c.seconds]
        rate = (tail[0]["z"] - tail[-1]["z"]) / max(1e-9, tail[-1]["t"] - tail[0]["t"])
        row = dict(
            dt=dt,
            sink_mm=(rig.box_z0 - rig.z) * 1e3,
            tail_rate_mm_s=rate * 1e3,
            f_sand=rig.log[-1]["f_sand"],
            weight=rig.mass * c.gravity,
            pen_mm=rig.log[-1]["pen"] * 1e3,
        )
        out.append(row)
        print(
            f"  dt={dt:<8.4f} sink {row['sink_mm']:8.2f} mm   tail rate {row['tail_rate_mm_s']:8.3f} mm/s   "
            f"F {row['f_sand']:9.2f} N / W {row['weight']:9.2f} N   maxpen {row['pen_mm']:7.3f} mm"
        )
    if len(out) >= 2:
        r = out[0]["sink_mm"] / max(1e-12, out[-1]["sink_mm"])
        span = dts[0] / dts[-1]
        print()
        print(f"  sink(dt={dts[0]}) / sink(dt={dts[-1]}) = {r:.3f}   (dt ratio {span:.1f})")
        print(f"  ~1.0  => (b) position drift      ~{1 / span:.2f} => (c) per-step leak")
    print()
    return out


def test_hold(cfg: Cfg, overlaps_mm=(0.0, 2.0, 5.0, 10.0, 20.0, 40.0)):
    print("=" * 78)
    print("TEST hold -- KINEMATIC box at a fixed overlap. Separates (b) from (a).")
    print("=" * 78)
    print("  contact_solver_kernels.py:65 solve_coulomb_isotropic:")
    print("      u_n = dot(u, nor); if u_n < 0: u -= u_n*nor")
    print("  i.e. the target normal relative velocity is EXACTLY 0, with no gap, no")
    print("  penetration-depth bias, no Baumgarte term. rasterized_collisions.py:488")
    print("  writes the node sdf and nothing reads it.")
    print("  PREDICTION  position-level law : reaction grows (roughly linearly) with overlap")
    print("              velocity-level law : reaction is INDEPENDENT of overlap, and equals")
    print("                                   only what is needed to stop the incoming flow")
    out = []
    for ov in overlaps_mm:
        c = Cfg(**cfg.__dict__)
        rig = Rig(c, with_box=True, kinematic_box=True, box_gap=-ov * 1e-3)
        rig.z = c.bed_z + c.box_hz - ov * 1e-3
        rig.run(c.seconds, sample_every=0.5)
        fz = float(np.mean([r["f_sand"] for r in rig.log if r["t"] > 0.5 * c.seconds]))
        row = dict(
            overlap_mm=ov,
            Fz=fz,
            pen_mm=rig.log[-1]["pen"] * 1e3,
            surface=rig.log[-1]["surface"],
            weight=rig.mass * c.gravity,
        )
        out.append(row)
        print(f"  overlap {ov:6.1f} mm   steady Fz {fz:12.3f} N   maxpen {row['pen_mm']:8.3f} mm")
    if len(out) >= 2:
        f0 = out[1]["Fz"] if abs(out[0]["Fz"]) < 1e-6 else out[0]["Fz"]
        print()
        print(f"  Fz(40mm)/Fz(2mm) = {out[-1]['Fz'] / max(1e-9, abs(f0)):.3f}")
        print("  >> 1 => position-level;  ~1 => velocity-level (MECHANISM B)")
    print()
    return out


def test_iters(cfg: Cfg):
    print("=" * 78)
    print("TEST iters -- convergence sweep. Confirms or denies (c).")
    print("=" * 78)
    combos = [
        (24, 1e-4),
        (25, 1e-4),
        (50, 1e-4),
        (100, 1e-4),
        (500, 1e-4),
        (100, 1e-8),
        (500, 1e-8),
    ]
    out = []
    for it, tol in combos:
        c = Cfg(**{**cfg.__dict__, "max_iterations": it, "tolerance": tol})
        rig = Rig(c, with_box=True)
        rig.solver.verbose = True
        rig.run(c.seconds, sample_every=0.5)
        res = rig.log[-1]["res"]
        row = dict(
            max_iterations=it,
            tolerance=tol,
            sink_mm=(rig.box_z0 - rig.z) * 1e3,
            res_l2=res[0] if res else None,
            res_linf=res[1] if res else None,
            rheo_calls=rig.log[-1]["rheo_calls"],
            host_batches=TRACE.predicted_host_batches,
            granularity=TRACE.granularity,
            pen_mm=rig.log[-1]["pen"] * 1e3,
            f_sand=rig.log[-1]["f_sand"],
            weight=rig.mass * c.gravity,
        )
        out.append(row)
        print(
            f"  it={it:<5d} tol={tol:<8.0e} sink {row['sink_mm']:8.2f} mm  "
            f"res(l2)={row['res_l2']}  res(linf)={row['res_linf']}  "
            f"batches={row['host_batches']} gran={row['granularity']} calls={row['rheo_calls']}"
        )
    print()
    print("  Monotone fall in sinkage with iteration count => MECHANISM C.")
    print("  Flat sinkage with a residual that is already below tolerance => the")
    print("  residual is not measuring feasibility (see evaluate_strain_residual,")
    print("  rheology_solver_kernels.py:1477 -- it is |W.dsigma|, a STAGNATION")
    print("  measure, not the flow-rule / complementarity violation).")
    print()
    return out


def test_basis(cfg: Cfg):
    print("=" * 78)
    print("TEST basis -- strain x velocity space pairs (inf-sup / locking check)")
    print("=" * 78)
    print("  Config default strain_basis='P0' with velocity_basis='Q1' is the classic")
    print("  Q1/P0 pair, which fails the LBB condition in 3D and admits checkerboard")
    print("  pressure modes: pressures of arbitrary magnitude that lie in ker(B^T) and")
    print("  therefore exert NO net force. If the measured stress field is dominated by")
    print("  such a mode, it is huge, flat with depth, AND consistent with a correct")
    print("  harvested force. Refining the voxel does NOT cure an inf-sup failure, which")
    print("  is exactly what voxel 0.08 -> 0.04 showed.")
    # Each pair runs in its OWN PROCESS. An unsupported or broken pair raises
    # CUDA error 700 (illegal memory access), which poisons the CUDA context
    # and takes the interpreter with it -- a try/except here cannot survive
    # that, and the first bad pair used to destroy the whole table.
    import os
    import subprocess
    import tempfile

    print("  each pair in its own process: a bad pair raises CUDA 700, which is")
    print("  unrecoverable in-process and would otherwise kill the whole sweep.")
    out = []
    for sb in ("P0", "P1d", "Q1d", "Q1"):
        for vb in ("Q1", "B2"):
            with tempfile.TemporaryDirectory() as td:
                jf = os.path.join(td, "row.json")
                argv = [
                    sys.executable, os.path.abspath(__file__), "basis1",
                    "--strain-basis", sb, "--velocity-basis", vb,
                    "--seconds", str(cfg.seconds), "--dt", str(cfg.dt),
                    "--voxel", str(cfg.voxel_size), "--ppc", str(cfg.particles_per_cell),
                    "--max-iterations", str(cfg.max_iterations),
                    "--tolerance", str(cfg.tolerance),
                    "--density", str(cfg.density), "--young", str(cfg.young_modulus),
                    "--bearing-kpa", str(cfg.target_bearing_kpa),
                    "--bed", str(cfg.bed_x), str(cfg.bed_y), str(cfg.bed_z),
                    "--wall-mu", str(cfg.wall_mu),
                    "--collider-basis", cfg.collider_basis,
                    "--critical-fraction", str(cfg.critical_fraction),
                    "--json", jf,
                ]
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
                row = None
                if os.path.exists(jf):
                    try:
                        with open(jf) as fh:
                            row = json.load(fh).get("basis1")
                    except Exception:
                        row = None
                if not isinstance(row, dict):
                    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
                    cuda = [t for t in tail if "CUDA error" in t]
                    why = cuda[0][:120] if cuda else (tail[-1][:120] if tail else "no output")
                    row = dict(strain_basis=sb, velocity_basis=vb,
                               error=f"rc={proc.returncode}: {why}")
                out.append(row)
                print(f"  {sb:>4}/{vb:<3} {row}")
    print()
    print("  slope_ratio is the number that matters: 1.0 means the bed carries its")
    print("  own weight with sigma_zz = rho*g*z. P0/Q1 measured 0.379.")
    print()
    return out


def test_basis1(cfg: Cfg):
    """One basis pair, so the parent can run it in its own process.

    test_basis spawns this; run it directly with --strain-basis/--velocity-basis
    to check a single pair by hand.
    """
    rig = Rig(cfg, with_box=True)
    rig.run(cfg.seconds, sample_every=1.0)
    prof = rig.stress_profile()
    fit = prof["fit"] or {}
    row = dict(
        strain_basis=cfg.strain_basis,
        velocity_basis=cfg.velocity_basis,
        sink_mm=(rig.box_z0 - rig.z) * 1e3,
        slope_ratio=fit.get("slope_ratio"),
        r2=fit.get("r2"),
        p_mid=prof["rows"][len(prof["rows"]) // 2]["solver"]["p"] if prof["rows"] else None,
        f_sand=rig.log[-1]["f_sand"],
    )
    print(f"  {cfg.strain_basis:>4}/{cfg.velocity_basis:<3} {row}")
    return row


def test_project(cfg: Cfg):
    print("=" * 78)
    print("TEST project -- with vs without solver.project_outside()")
    print("=" * 78)
    print("  project_outside_collider (rasterized_collisions.py:347) is the ONLY")
    print("  position-level non-penetration correction in the whole solver, and")
    print("  solver_implicit_mpm.py never calls it from step(): it is reachable only")
    print("  through the public solver.project_outside() method (line 2205).")
    print("  example_mpm_granular / water_dam_break / multi_material / grain_rendering")
    print("  all call it every step. example_mpm_twoway_coupling.py does NOT -- and the")
    print("  coupled Isaac Lab entry cannot (project_outside_colliders=False).")
    out = []
    for on in (False, True):
        c = Cfg(**{**cfg.__dict__, "project_outside": on})
        rig = Rig(c, with_box=True)
        rig.run(c.seconds, sample_every=0.5)
        tail = [r for r in rig.log if r["t"] > 0.75 * c.seconds]
        rate = (tail[0]["z"] - tail[-1]["z"]) / max(1e-9, tail[-1]["t"] - tail[0]["t"])
        row = dict(
            project_outside=on,
            sink_mm=(rig.box_z0 - rig.z) * 1e3,
            tail_rate_mm_s=rate * 1e3,
            pen_mm=rig.log[-1]["pen"] * 1e3,
            f_sand=rig.log[-1]["f_sand"],
            weight=rig.mass * c.gravity,
        )
        out.append(row)
        print(
            f"  project_outside={on!s:<5} sink {row['sink_mm']:8.2f} mm  "
            f"tail rate {row['tail_rate_mm_s']:7.3f} mm/s  maxpen {row['pen_mm']:7.3f} mm  "
            f"F {row['f_sand']:8.2f}/{row['weight']:8.2f} N"
        )
    print()
    return out


# ---------------------------------------------------------------------------


TESTS = {
    "env": test_env,
    "geostatic": test_geostatic,
    "dt": test_dt,
    "hold": test_hold,
    "iters": test_iters,
    "basis": test_basis,
    "basis1": test_basis1,
    "project": test_project,
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test", nargs="?", default="all", choices=[*TESTS, "all"])
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--voxel", type=float, default=0.08)
    p.add_argument("--ppc", type=int, default=8)
    p.add_argument("--max-iterations", type=int, default=24)
    p.add_argument("--tolerance", type=float, default=1.0e-4)
    p.add_argument("--density", type=float, default=6000.0)
    p.add_argument("--young", type=float, default=1.0e8)
    p.add_argument("--bearing-kpa", type=float, default=10.0)
    p.add_argument("--bed", type=float, nargs=3, default=(0.60, 0.60, 0.24), metavar=("X", "Y", "Z"))
    p.add_argument("--wall-mu", type=float, default=0.0)
    p.add_argument("--strain-basis", default="P0")
    p.add_argument("--velocity-basis", default="Q1")
    p.add_argument("--collider-basis", default="pic27")
    p.add_argument("--critical-fraction", type=float, default=0.90)
    p.add_argument("--host-loop", action="store_true", help="force the non-CUDA-graph solver loop")
    p.add_argument("--verbose-solver", action="store_true")
    p.add_argument("--no-subtract", action="store_true", help="skip the subtract_body_force step")
    p.add_argument("--device", default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)

    wp.init()
    install_instrumentation()

    cfg = Cfg(
        dt=a.dt,
        seconds=a.seconds,
        voxel_size=a.voxel,
        particles_per_cell=a.ppc,
        max_iterations=a.max_iterations,
        tolerance=a.tolerance,
        density=a.density,
        young_modulus=a.young,
        target_bearing_kpa=a.bearing_kpa,
        bed_x=a.bed[0],
        bed_y=a.bed[1],
        bed_z=a.bed[2],
        wall_mu=a.wall_mu,
        strain_basis=a.strain_basis,
        velocity_basis=a.velocity_basis,
        collider_basis=a.collider_basis,
        critical_fraction=a.critical_fraction,
        host_loop=a.host_loop,
        verbose_solver=a.verbose_solver,
        subtract_body_force=not a.no_subtract,
    )

    print(f"# bed {cfg.bed_x}x{cfg.bed_y}x{cfg.bed_z} m, box {2 * cfg.box_hx}x{2 * cfg.box_hy}x{2 * cfg.box_hz} m, "
          f"mass {cfg.derived_box_mass():.1f} kg, bearing {cfg.bearing_kpa():.2f} kPa, g {cfg.gravity} m/s^2")
    print(f"# host_loop={cfg.host_loop} max_iterations={cfg.max_iterations} tolerance={cfg.tolerance}")
    print()

    names = list(TESTS) if a.test == "all" else [a.test]
    results = {}
    ctx = wp.ScopedDevice(a.device) if a.device else None
    try:
        if ctx:
            ctx.__enter__()
        for n in names:
            try:
                results[n] = TESTS[n](cfg)
            except Exception as e:
                import traceback

                traceback.print_exc()
                results[n] = dict(error=repr(e))
    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"# wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
