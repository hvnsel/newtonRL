# agents/rsl_rl_ppo_cfg.py
#
# Isaac Lab converts this "policy" style config into rsl_rl's newer actor /
# critic model configs at start-up (isaaclab_rl.rsl_rl.utils), so it works
# with the rsl_rl version the develop branch installs.

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class TricyclePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 50          # half an episode per rollout at 50 Hz
    max_iterations = 300
    save_interval = 50
    experiment_name = "tricycle"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=4,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
