# agents/rsl_rl_ppo_cfg.py
#
# Same shape as the navigate runner, with a longer rollout. The MPM tier runs
# tens of envs, so the batch comes from rollout length: 16 envs x 256 steps is
# 4096 samples per update and 1024 per minibatch.
#
# gamma 0.997 is an effective horizon of 333 steps, 13 s at 25 Hz, against a
# 500-step episode and a cut that runs 10-20 s.

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ExcavatorExcavatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 256
    max_iterations = 3000
    save_interval = 100
    experiment_name = "excavator_excavate"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    clip_actions = 1.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
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
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.997,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
