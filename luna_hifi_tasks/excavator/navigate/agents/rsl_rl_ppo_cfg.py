# agents/rsl_rl_ppo_cfg.py
#
# Same "policy" style config as the tricycle, which Isaac Lab converts into
# rsl_rl's model configs at start-up (isaaclab_rl.rsl_rl.utils).
#
# Two things differ from the tricycle and both matter:
#   * observation normalisation is ON. 215 inputs spanning body velocities,
#     unit vectors and metres of terrain relief do not share a scale.
#   * obs_groups routes the env's "critic" observation group to the critic
#     only. That is the asymmetric actor-critic: the critic sees the clean
#     scan and true slip, the actor sees what a rover could sense.
#
# gamma 0.995 is an effective horizon of 200 steps, 8 s at 25 Hz, against a
# 500-step episode. A goal reached 2.5 s out is discounted to 0.73 rather
# than the 0.29 that gamma 0.99 at 50 Hz gave.

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ExcavatorNavigatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 100
    experiment_name = "excavator_navigate"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    clip_actions = 1.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
