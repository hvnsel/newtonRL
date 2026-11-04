# PPO runner config for the excavate task.

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ExcavatorExcavatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Steps per env per update; an episode is 600 steps, the first 75 settling.
    num_steps_per_env = 256
    max_iterations = 800
    save_interval = 5
    experiment_name = "excavator_excavate"
    empirical_normalization = False
    # The critic also reads the privileged "critic" group.
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
        schedule="fixed",
        gamma=0.997,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
