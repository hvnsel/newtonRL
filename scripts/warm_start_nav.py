"""Widen a navigate checkpoint trained blind to the current observation
layout, so training can start from it.

    ~/IsaacLab/.venv/bin/python scripts/warm_start_nav.py SRC DST

SRC is a checkpoint whose actor reads the 23 inputs navigate had before
goal_vec_near, trained with env.actor_scan=false: model_1199. DST reads the
current layout, goal_vec_near and both terrain scans included, 249 inputs for
the actor and 480 for the critic, which reads the policy group plus its own
231.

Each old column is copied to where the same term now sits, by name, and every
new column gets zero weight, so at iteration 0 DST acts exactly as SRC does
and only starts weighting the new inputs where the gradient says to. The
critic is widened the same way.

The observation normaliser keeps SRC's statistics for the old columns. The
new ones get the priors in NEW_STATS: SRC's running count is in the tens of
millions, so the normaliser would barely move off whatever they start at.

DST keeps SRC's action std, starts at iteration 0, and has no optimizer state,
since every first-layer shape changed. Train from it with

    workstation/train.sh train --rl_library rsl_rl --task Luna-Excavator-Navigate \\
        --num_envs 1024 --checkpoint DST --no-resume_optimizer_state

Without --no-resume_optimizer_state the load stops on the missing optimizer
state instead of failing mid-run on mismatched shapes.

Needs torch, so run it with the Isaac Lab venv's python. It reads the layout
from luna_hifi_tasks/excavator/mdp/observations.py by file path, since the
package's __init__ imports Isaac Lab.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
OBS_PY = REPO / "luna_hifi_tasks" / "excavator" / "mdp" / "observations.py"

# SRC's actor input, in order: navigate_obs_spec(include_scan=False) as it
# stood before goal_vec_near was added.
OLD_POLICY = [
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("goal_vec_b", 2),
    ("goal_heading", 2),
    ("wheel_vel", 4),
    ("arm_pos", 2),
    ("drum_fill", 2),
    ("last_action", 2),
]

# (mean, std) for the columns SRC never saw. The scans read chassis height
# above the ground, 0.30 m on the flat with the wheels' 0.30 m radius; a std of
# 0.5 keeps a 0.5 m boulder near -1, a crater near +1.4 and a dropped cell's
# -2.0 near -4.6. goal_vec_near is a vector of length at most 1.
NEW_STATS = {
    "goal_vec_near": (0.0, 0.5),
    "far_scan": (0.30, 0.5),
    "near_scan": (0.30, 0.5),
}

FIRST = "mlp.0.weight"
NORM = "obs_normalizer."


def _load_observations():
    spec = importlib.util.spec_from_file_location("_nav_observations", OBS_PY)
    mod = importlib.util.module_from_spec(spec)
    # dataclass resolves its module through sys.modules
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _columns(terms: list[tuple[str, int]]) -> tuple[dict[str, tuple[int, int]], int]:
    out, off = {}, 0
    for name, dim in terms:
        if name in out:
            raise SystemExit(f"duplicate observation term {name!r}")
        out[name] = (off, off + dim)
        off += dim
    return out, off


def _first_layer_gap(w, mean, std, w_new, mean_new, std_new, old_cols, new_cols) -> tuple[float, float]:
    """How far DST's first layer is from SRC's: the largest difference in
    pre-activation over random inputs that agree on the old columns, whatever
    the new ones hold, and the largest sum of absolute terms that difference
    sits in. mean and std are None without a normaliser.

    In float64. A trained normaliser divides an input that never varies --
    the held arm, an always-empty drum -- by its std plus 0.01, so a random
    test value there lands near a hundred and a row sums to hundreds; float32
    rounds a 254-wide and a 480-wide sum of that size apart by ~1e-4, which
    no fixed tolerance tells from a misplaced column. float64 rounds them
    apart by ~1e-16 of the sum.
    """
    f64 = torch.float64
    old_dim, new_dim = w.shape[1], w_new.shape[1]
    x_old = torch.randn(64, old_dim, dtype=f64)
    x_new = torch.randn(64, new_dim, dtype=f64) * 3.0
    for name, (oa, ob) in old_cols.items():
        a, b = new_cols[name]
        x_new[:, a:b] = x_old[:, oa:ob]
    if mean is not None:
        eps = 1e-2      # EmpiricalNormalization's default
        x_old = (x_old - mean.reshape(1, -1).to(f64)) / (std.reshape(1, -1).to(f64) + eps)
        x_new = (x_new - mean_new.reshape(1, -1).to(f64)) / (std_new.reshape(1, -1).to(f64) + eps)
    w64, w_new64 = w.to(f64), w_new.to(f64)
    gap = float((x_old @ w64.T - x_new @ w_new64.T).abs().max())
    scale = max(float((x_old.abs() @ w64.abs().T).max()), 1e-30)
    return gap, scale


def _widen(sd: dict, old_terms, new_terms, label: str) -> list[str]:
    """Rewrite the first layer and the normaliser of one model in place.
    Returns the names of the new terms."""
    old_cols, old_dim = _columns(old_terms)
    new_cols, new_dim = _columns(new_terms)

    if FIRST not in sd:
        raise SystemExit(f"{label}: no {FIRST!r} in the checkpoint; keys are {sorted(sd)[:8]} ...")
    w = sd[FIRST]
    if w.shape[1] == new_dim:
        raise SystemExit(f"{label}: already reads {new_dim} inputs; SRC looks widened already")
    if w.shape[1] != old_dim:
        raise SystemExit(
            f"{label}: first layer reads {w.shape[1]} inputs, expected {old_dim} for the "
            f"blind layout {[n for n, _ in old_terms]}"
        )
    dropped = sorted(set(old_cols) - set(new_cols))
    if dropped:
        raise SystemExit(f"{label}: terms in SRC with no place in the new layout: {dropped}")

    w_new = w.new_zeros(w.shape[0], new_dim)
    has_norm = f"{NORM}_mean" in sd
    if has_norm:
        mean, var, std = (sd[f"{NORM}{k}"] for k in ("_mean", "_var", "_std"))
        if mean.shape[-1] != old_dim:
            raise SystemExit(f"{label}: normaliser covers {mean.shape[-1]} inputs, expected {old_dim}")
        mean_new = mean.new_zeros(1, new_dim)
        var_new = var.new_ones(1, new_dim)
        std_new = std.new_ones(1, new_dim)

    added = []
    for name, (a, b) in new_cols.items():
        if name in old_cols:
            oa, ob = old_cols[name]
            if ob - oa != b - a:
                raise SystemExit(f"{label}: {name} was {ob - oa} wide and is now {b - a}")
            w_new[:, a:b] = w[:, oa:ob]
            if has_norm:
                mean_new[:, a:b] = mean.reshape(1, -1)[:, oa:ob]
                var_new[:, a:b] = var.reshape(1, -1)[:, oa:ob]
                std_new[:, a:b] = std.reshape(1, -1)[:, oa:ob]
        else:
            if name not in NEW_STATS:
                raise SystemExit(f"{label}: new term {name!r} has no prior in NEW_STATS")
            m, s = NEW_STATS[name]
            if has_norm:
                mean_new[:, a:b] = m
                var_new[:, a:b] = s * s
                std_new[:, a:b] = s
            added.append(name)

    # The point of the zero columns: the first layer's pre-activation is
    # unchanged for any input that agrees with SRC's on the old columns,
    # whatever the new ones hold.
    gap, scale = _first_layer_gap(
        w, mean if has_norm else None, std if has_norm else None,
        w_new, mean_new if has_norm else None, std_new if has_norm else None,
        old_cols, new_cols,
    )
    if gap > 1e-9 * scale:
        raise SystemExit(
            f"{label}: widened first layer does not reproduce SRC's: off by {gap:.3g} "
            f"in sums of up to {scale:.3g}"
        )

    sd[FIRST] = w_new
    if has_norm:
        sd[f"{NORM}_mean"] = mean_new.reshape(mean.shape[:-1] + (new_dim,))
        sd[f"{NORM}_var"] = var_new.reshape(var.shape[:-1] + (new_dim,))
        sd[f"{NORM}_std"] = std_new.reshape(std.shape[:-1] + (new_dim,))
    count = int(sd[f"{NORM}count"]) if f"{NORM}count" in sd else None
    print(f"[warm start] {label:<6} {old_dim:>4} -> {new_dim:>4} inputs, "
          f"{sum(new_cols[n][1] - new_cols[n][0] for n in added)} new at zero weight "
          f"({', '.join(added)})"
          + (f"; normaliser count {count:,}" if count is not None else "; no normaliser")
          + f"; matches SRC to {gap / scale:.0e}")
    return added


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", help="checkpoint trained on the 23-input blind layout (model_1199)")
    p.add_argument("dst", help="where to write the widened checkpoint")
    args = p.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_file():
        raise SystemExit(f"no checkpoint at {src}")
    if dst.resolve() == src.resolve():
        raise SystemExit("DST is SRC; write the widened checkpoint somewhere else")

    obs = _load_observations()
    new_policy = [(t.name, t.dim) for t in obs.navigate_obs_spec(include_scan=True).terms]
    critic_state = [(t.name, t.dim) for t in obs.critic_state_spec(terrain_cells=obs.NAV_SCAN_CELLS).terms]
    clash = {n for n, _ in critic_state} & ({n for n, _ in new_policy} | {n for n, _ in OLD_POLICY})
    if clash:
        raise SystemExit(f"critic and policy terms share names: {sorted(clash)}")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    for key in ("actor_state_dict", "critic_state_dict"):
        if key not in ckpt:
            raise SystemExit(f"{src} has no {key}; keys are {sorted(ckpt)}")

    _widen(ckpt["actor_state_dict"], OLD_POLICY, new_policy, "actor")
    # The critic reads the policy group, then its own: obs_groups
    # {"critic": ["policy", "critic"]} in the PPO cfg.
    _widen(ckpt["critic_state_dict"], OLD_POLICY + critic_state, new_policy + critic_state, "critic")

    actor = ckpt["actor_state_dict"]
    if "distribution.std_param" in actor:
        std = actor["distribution.std_param"]
    elif "distribution.log_std_param" in actor:
        std = actor["distribution.log_std_param"].exp()
    else:
        std = None
    if std is not None:
        print(f"[warm start] action std kept: forward {float(std[0]):.3f}, yaw {float(std[1]):.3f}")

    print(f"[warm start] iteration {ckpt.get('iter')} -> 0; optimizer state dropped")
    ckpt["iter"] = 0
    ckpt.pop("optimizer_state_dict", None)
    ckpt.setdefault("infos", None)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dst)
    print(f"[warm start] wrote {dst}")
    print("train from it with:")
    print("  workstation/train.sh train --rl_library rsl_rl --task Luna-Excavator-Navigate "
          f"--num_envs 1024 --checkpoint {dst} --no-resume_optimizer_state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
