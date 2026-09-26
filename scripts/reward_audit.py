# scripts/reward_audit.py
#
# Run a task under fixed policies and print what each reward term is worth
# over a whole episode.
#
#   isaaclab -p scripts/reward_audit.py --task Luna-Excavator-Navigate --episodes 200
#   isaaclab -p scripts/reward_audit.py --task Luna-Excavator-Excavate --episodes 40
#
# What a policy optimises is a term's episode total, which is its weight
# times its typical magnitude times the steps it is paid over. Two columns are
# printed:
#
#   zero    every action held at 0, the null baseline
#   fresh   a new draw from N(0, init_noise_std) every step, which is what an
#           untrained PPO policy emits
#
# The std is read from the task's own rsl_rl config, so the action deltas the
# audit measures are the ones training will see. The env's action_smoothing
# applies on top, the same for both.
#
# Terms come from the env's own TermLogger, already signed and weighted, so
# their sum is the episode return.

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

POLICIES = ("zero", "fresh")


def _parse(argv):
    p = argparse.ArgumentParser(description="Measure per-term episode reward totals.")
    p.add_argument("--task", required=True,
                   help="Luna-Excavator-Navigate or Luna-Excavator-Excavate")
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--episodes", type=int, default=4,
                   help="episodes for the fresh pass, rounded up to a whole "
                        "number of num_envs. The zero pass is deterministic "
                        "and always runs one round")
    p.add_argument("--action_std", type=float, default=None,
                   help="std the fresh pass samples at; defaults to the task's "
                        "own policy.init_noise_std")
    p.add_argument("--max_steps", type=int, default=200_000,
                   help="give up after this many env steps per policy")
    add_launcher_args(p)
    # visualizer_cfgs is empty on this path, so the named type selects
    # nothing and no window opens. "none" is not a type this build accepts.
    p.set_defaults(device=None, visualizer=["newton_gl"])
    args, hydra_args = setup_preset_cli(p, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    return args


def _policy_std(agent_cfg, override: float | None) -> tuple[float, str]:
    """Action std for the fresh pass, from the task's own rsl_rl config.

    An untrained PPO policy draws from N(mean, init_noise_std) independently
    each step, so its per-step action delta has std sqrt(2) * init_noise_std.
    Measuring against anything else misreports every term that reads an action
    difference.
    """
    if override is not None:
        return override, "--action_std"
    std = getattr(getattr(agent_cfg, "policy", None), "init_noise_std", None)
    if std is None and isinstance(agent_cfg, dict):
        std = agent_cfg.get("policy", {}).get("init_noise_std")
    if std is None:
        raise SystemExit(
            f"no policy.init_noise_std on the rsl_rl config "
            f"({type(agent_cfg).__name__}); pass --action_std explicitly."
        )
    return float(std), "policy.init_noise_std"


def _collect(env, kind: str, episodes: int, std: float, max_steps: int):
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
        if kind == "fresh":
            action.normal_(0.0, std).clamp_(-1.0, 1.0)
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
        print(f"\r  {kind}: {seen}/{episodes} episodes, {steps} steps", end="", flush=True)

    print()
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

    # The verdict is on signal share: how much the terms that pay for doing
    # the task come to next to the ones that charge for trying. A fresh policy
    # scoring below a stationary one is expected; a fresh policy whose
    # positives are invisible next to its penalties is what training has to
    # climb out of.
    pos = {k: sum(v for v in results[k][0].values() if v > 0) for k in results}
    neg = {k: -sum(v for v in results[k][0].values() if v < 0) for k in results}
    share = pos["fresh"] / max(neg["fresh"], 1e-9)
    print()
    if share < 0.10:
        print(f"  BROKEN: a fresh policy earns {pos['fresh']:.2f} against "
              f"{neg['fresh']:.2f} in penalties ({share:.0%}).")
        print("          The signal is buried; scale the dominant penalty down.")
    else:
        print(f"  OK: a fresh policy earns {pos['fresh']:.2f} against "
              f"{neg['fresh']:.2f} in penalties ({share:.0%}). The signal is visible.")

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

    # One Hydra pass for both; the agent cfg is what carries init_noise_std.
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    std, std_src = _policy_std(agent_cfg, args.action_std)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
        # The sparse-grid capacities are absolute totals sized from
        # max_num_envs, and __post_init__ is what derives them, so both are
        # set to the run's env count and it is run again.
        if hasattr(env_cfg, "max_num_envs"):
            env_cfg.max_num_envs = args.num_envs
            env_cfg.__post_init__()
    if hasattr(env_cfg, "apply_soil_material"):
        env_cfg.apply_soil_material()

    args.device = env_cfg.sim.device
    env_cfg.validate()

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            u = env.unwrapped
            ep_len = int(u.max_episode_length)
            # zero holds every action at 0, so every episode of it is the
            # same episode and one round of envs is the whole sample.
            # --episodes sizes the fresh pass.
            plan = {"zero": u.num_envs, "fresh": max(args.episodes, u.num_envs)}
            rounds = sum(-(-n // u.num_envs) for n in plan.values())
            print(f"[audit] {args.task}  {u.num_envs} envs  {ep_len} steps/episode  "
                  f"{1.0 / (env_cfg.sim.dt * env_cfg.decimation):.0f} Hz")
            print(f"[audit] zero 1 round, fresh {-(-plan['fresh'] // u.num_envs)} rounds "
                  f"-> {rounds * ep_len:,} env-steps total")
            print(f"[audit] fresh samples N(0, {std:.2f}) per step from {std_src}; "
                  f"action delta std {std * 2 ** 0.5:.2f}, "
                  f"smoothed by {u.cfg.action_smoothing:.2f}")
            results = {}
            for kind in POLICIES:
                print(f"[audit] {kind} "
                      + ("(actions held at 0 -- nothing should move)" if kind == "zero"
                         else f"(new N(0, {std:.2f}) draw every step)"), flush=True)
                results[kind] = _collect(
                    env, kind, plan[kind], std, args.max_steps
                )
            _report(results)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
