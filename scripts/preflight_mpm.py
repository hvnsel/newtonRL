"""Test 0. Costs one second and can invalidate every measurement taken so far.

If wp.is_conditional_graph_supported() is False, SolverImplicitMPM takes the host
branch of _run_solver_loop, which is

    for batch in range(max_iterations // solve_granularity):

with _GaussSeidelSolver.solve_granularity == 25. At max_iterations=24 that is
range(0): ZERO rheology sweeps and ZERO contact sweeps per step. The step then
advances on the warm start alone and the constitutive model is bypassed entirely.
"""
import warp as wp

wp.init()
ok = wp.is_conditional_graph_supported()
gran, it = 25, 24
print(f"wp.is_conditional_graph_supported() = {ok}")
print(f"device = {wp.get_device()}")
if ok:
    print(f"  -> GRAPH path: solve_granularity=5, iteration starts at 1,")
    print(f"     cur_it = 6,11,16,21,26; stops when 26 > {it}")
    print(f"     => {5 * 5} Gauss-Seidel sweeps per step")
    print(f"     Needed: L/h = 5.8/0.08 = 72 sweeps to transport the load path,")
    print(f"     and c*dt/h = sqrt(1e8/6000)*0.01/0.08 = 16 cells/step elastic CFL.")
else:
    print(f"  -> HOST path: range({it} // {gran}) = range({it // gran})")
    if it // gran == 0:
        print("     *** ZERO SWEEPS PER STEP. The rheology solver never runs. ***")
        print("     *** Every prior measurement was taken on a solver that")
        print("     *** never iterated. Fix max_iterations before anything else.")
