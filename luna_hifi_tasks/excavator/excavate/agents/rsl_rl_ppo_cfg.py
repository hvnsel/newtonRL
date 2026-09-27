# agents/rsl_rl_ppo_cfg.py

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ExcavatorExcavatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 256 # Number of steps to run per environment before updating the policy
    max_iterations = 3000 # Maximum number of training iterations to run
    save_interval = 100 # Interval at which to save the model and training state
    experiment_name = "excavator_excavate"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]} # the critic has privileged access to observations not available to the actor
    clip_actions = 1.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64], # Hidden layer sizes for the actor network
        critic_hidden_dims=[256, 128, 64], # Hidden layer sizes for the critic network
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
