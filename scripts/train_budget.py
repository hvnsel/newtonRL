# scripts/train_budget.py
#
# Turn a MEASURED throughput into GPU hours, and hours into a credit estimate.
#
#   python scripts/train_budget.py --sps 420 --envs 1
#   python scripts/train_budget.py --sps 420 --envs 1 --target-envs 64 --rate 1.0
#
# Every training estimate for this project is
#
#     hours = env-steps needed / env-steps per second / 3600
#
# and the second term is the only one that needs a machine to find out. Run
# dig_demo (or smoke_test) once on the hardware in question, take the
# env-steps/s it prints, and put it in here. Everything else is arithmetic and
# published sample budgets.
#
# The scaling to more envs is DELIBERATELY optimistic-with-a-caveat: MPM envs
# do not scale linearly, because the sparse grid and the coupler are shared
# work. Treat --target-envs as an upper bound and re-measure at the real env
# count before committing to a long job.

from __future__ import annotations

import argparse

# Rough published sample budgets for PPO on comparable tasks. These are order
# of magnitude, not promises.
BUDGETS = {
    "navigate": (50_000_000, 200_000_000),
    "excavate": (10_000_000, 50_000_000),
    "planner": (2_000_000, 10_000_000),
}


def main() -> int:
    p = argparse.ArgumentParser(description="GPU hours from measured throughput.")
    p.add_argument("--sps", type=float, required=True,
                   help="MEASURED env-steps/s, from dig_demo's throughput line")
    p.add_argument("--envs", type=int, default=1, help="env count that was measured")
    p.add_argument("--target-envs", type=int, default=None,
                   help="env count you intend to train at (default: as measured)")
    p.add_argument("--rate", type=float, default=None,
                   help="credits per GPU-hour, if you know your site's charge rate")
    p.add_argument("--budget", type=float, default=None, help="credits available")
    args = p.parse_args()

    target = args.target_envs or args.envs
    per_env = args.sps / max(args.envs, 1)
    # Linear in env count, then discounted: shared grid and coupler work does
    # not parallelise perfectly across envs.
    efficiency = 1.0 if target <= args.envs else 0.6
    sps = per_env * target * efficiency

    print(f"measured {args.sps:,.0f} env-steps/s over {args.envs} env "
          f"= {per_env:,.1f} per env")
    if target != args.envs:
        print(f"projected to {target} envs at {efficiency:.0%} scaling efficiency "
              f"-> {sps:,.0f} env-steps/s")
        print("  (a projection, not a measurement -- re-measure at the real env count)")
    print()

    print(f"{'policy':<10}{'steps (low)':>14}{'hours':>8}{'steps (high)':>15}{'hours':>8}")
    total_lo = total_hi = 0.0
    for name, (lo, hi) in BUDGETS.items():
        h_lo, h_hi = lo / sps / 3600, hi / sps / 3600
        total_lo, total_hi = total_lo + h_lo, total_hi + h_hi
        print(f"{name:<10}{lo:>14,}{h_lo:>8.1f}{hi:>15,}{h_hi:>8.1f}")
    print(f"{'TOTAL':<10}{'':>14}{total_lo:>8.1f}{'':>15}{total_hi:>8.1f}  GPU-hours")

    if args.rate:
        print(f"\nat {args.rate:g} credits/GPU-hour: "
              f"{total_lo * args.rate:,.0f} - {total_hi * args.rate:,.0f} credits")
        if args.budget:
            print(f"you have {args.budget:,.0f}: "
                  + ("comfortable" if total_hi * args.rate < args.budget * 0.5
                     else "tight" if total_lo * args.rate < args.budget
                     else "NOT ENOUGH at the high estimate"))
    else:
        print("\npass --rate to price it. Your site's charge rate per GPU-hour is a")
        print("published number and worth looking up before planning around 272 credits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
