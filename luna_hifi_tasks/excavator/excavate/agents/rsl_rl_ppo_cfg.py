# agents/rsl_rl_ppo_cfg.py

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ExcavatorExcavatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Steps per environment per policy update.
    #
    # An episode is episode_length_s / (sim.dt * decimation) = 24.0 / 0.04 =
    # 600 steps, of which the first settle_seconds = 3.0 s -- 75 steps -- are
    # unpaid settling with every progress delta zeroed while the bed stops
    # moving. At 64 this was shorter than the settle, so a rollout could
    # consist of nothing but settling and carry no commanded cut whatsoever,
    # and with gamma 0.997 (a ~333 step horizon) there was nothing for GAE to
    # propagate. 256 clears the settle and covers 43% of an episode.
    #
    # It also sets the batch. Dig runs at a handful of environments -- the MPM
    # sparse grid is the memory ceiling, not particle count -- so 4 x 64 = 256
    # samples split 4 ways left 64 per minibatch. 4 x 256 = 1024 gives 256.
    num_steps_per_env = 256
    max_iterations = 800 # Maximum number of training iterations to run
    # Checkpoints are one small MLP each, and an iteration is tens of minutes
    # on the dig task, so save often enough that a crash costs minutes rather
    # than hours.
    save_interval = 5
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
        schedule="fixed",
        gamma=0.997,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
