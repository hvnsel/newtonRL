# scripts/reward_audit.py
#
# Run a task under fixed policies and print what each reward term is worth
# over a whole episode.
#
#   isaaclab -p scripts/reward_audit.py --task Luna-Excavator-Navigate --episodes 200
#   isaaclab -p scripts/reward_audit.py --task Luna-Excavator-Excavate --episodes 40
#
# A reward weight on its own says nothing. What a policy optimises is the sum
# over an episode, which depends on the weight, on the typical magnitude of
# the term, on how many steps it is paid over, and on whether the episode ends
# early. Two columns are printed:
#
#   zero   every action held at 0
#   walk   a smoothed random walk over the action space
#
# `zero` is the null baseline. If it scores at or above `walk`, doing nothing
# beats attempting the task and no amount of training fixes that.
#
# Terms come from the env's own TermLogger, so they are already signed and
# weighted and their sum is the episode return.

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import setup_preset_cli
from isaaclab_tasks.utils.hydra import resolve_task_config

import luna_hifi_tasks  # noqa: F401  (registers Luna-* tasks)

POLICIES = ("zero", "walk")


def _parse(argv):
    p = argparse.ArgumentParser(description="Measure per-term episode reward totals.")
    p.add_argument("--task", required=True,
                   help="Luna-Excavator-Navigate or Luna-Excavator-Excavate")
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--episodes", type=int, default=100,
                   help="episodes to finish per policy")
    p.add_argument("--walk_sigma", type=float, default=0.15,
                   help="per-step action noise for the walk policy")
    p.add_argument("--max_steps", type=int, default=200_000,
                   help="give up after this many env steps per policy")
    add_launcher_args(p)
    p.set_defaults(device=None, visualizer=["none"])
    args, hydra_args = setup_preset_cli(p, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def _collect(env, kind: str, episodes: int, sigma: float, max_steps: int):
    """Run one policy until `episodes` have finished.

    Returns (per-term mean episode totals, episodes seen, mean episode length).
    """
    u = env.unwrapped
    n_act = env.action_space.shape[-1]
    action = torch.zeros(u.num_envs, n_act, device=u.device)

    env.reset()
    totals: dict[str, float] = {}
    seen = 0
    steps = 0

    while seen < episodes and steps < max_steps:
        if kind == "walk":
            action.add_(torch.randn_like(action) * sigma).clamp_(-1.0, 1.0)
        _, _, terminated, truncated, extras = env.step(action)
        steps += 1

        done = terminated | truncated
        k = int(done.sum())
        if k == 0:
            continue
        log = extras.get("log", {})
        for key, value in log.items():
            if not key.startswith("Episode_Reward/"):
                continue
            totals[key[len("Episode_Reward/"):]] = totals.get(key[len("Episode_Reward/"):], 0.0) + value * k
        seen += k
        # The walk keeps its state across an episode boundary otherwise, so a
        # fresh episode would start wherever the last one left the action.
        if kind == "walk":
            action[done] = 0.0

    if seen == 0:
        raise RuntimeError(
            f"no episode finished in {steps} steps under the {kind!r} policy; "
            "raise --max_steps or shorten episode_length_s"
        )
    return ({k: v / seen for k, v in totals.items()},
            seen,
            steps * u.num_envs / seen)


def _report(results: dict[str, tuple[dict[str, float], int, float]]) -> None:
    names = sorted(
        {n for r, _, _ in results.values() for n in r},
        key=lambda n: -max(abs(r.get(n, 0.0)) for r, _, _ in results.values()),
    )
    head = "  ".join(f"{k:>12}" for k in results)
    print(f"\n{'term':<16}{head}")
    print("-" * (16 + 14 * len(results)))
    for n in names:
        row = "  ".join(f"{results[k][0].get(n, 0.0):>12.3f}" for k in results)
        print(f"{n:<16}{row}")
    print("-" * (16 + 14 * len(results)))
    print(f"{'RETURN':<16}" + "  ".join(
        f"{sum(results[k][0].values()):>12.3f}" for k in results))
    print(f"{'episodes':<16}" + "  ".join(f"{results[k][1]:>12d}" for k in results))
    print(f"{'mean length':<16}" + "  ".join(f"{results[k][2]:>12.0f}" for k in results))

    zero, walk = (sum(results[k][0].values()) for k in ("zero", "walk"))
    print()
    if zero >= walk:
        print(f"  BROKEN: doing nothing scores {zero:.2f} against {walk:.2f} for moving.")
        print("          No amount of training fixes a reward whose optimum is idling.")
    else:
        print(f"  OK: moving scores {walk:.2f} against {zero:.2f} for doing nothing.")

    for kind, (terms, _, _) in results.items():
        pos = {n: v for n, v in terms.items() if v > 0}
        neg = {n: v for n, v in terms.items() if v < 0}
        big_p = max(pos.items(), key=lambda kv: kv[1], default=("-", 0.0))
        big_n = min(neg.items(), key=lambda kv: kv[1], default=("-", 0.0))
        print(f"  {kind:<5} largest reward {big_p[0]} {big_p[1]:+.2f}, "
              f"largest penalty {big_n[0]} {big_n[1]:+.2f}, "
              f"total {sum(pos.values()):+.2f} / {sum(neg.values()):+.2f}")


def main(argv=None) -> int:
    args = _parse(argv)
    torch.manual_seed(0)

    from luna_hifi_tasks.excavator.excavator_cfg import usd_status

    problem = usd_status()
    if problem is not None:
        print(f"\n[asset] {problem}\n")
        return 2

    env_cfg, _ = resolve_task_config(args.task, "")
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if hasattr(env_cfg, "apply_soil_material"):
        env_cfg.apply_soil_material()

    args.device = env_cfg.sim.device
    env_cfg.validate()

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            u = env.unwrapped
            print(f"[audit] {args.task}  {u.num_envs} envs  "
                  f"{u.max_episode_length} steps/episode  "
                  f"{1.0 / (env_cfg.sim.dt * env_cfg.decimation):.0f} Hz")
            results = {}
            for kind in POLICIES:
                print(f"[audit] {kind} ...", flush=True)
                results[kind] = _collect(
                    env, kind, args.episodes, args.walk_sigma, args.max_steps
                )
            _report(results)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
