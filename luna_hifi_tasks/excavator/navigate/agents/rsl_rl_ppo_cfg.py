# PPO for navigate (rsl_rl 5).
#
# Actor: MLP over the 25-wide "policy" group, normalised.
# Critic: MLP over the "policy" and privileged "critic" groups.
# Symmetry: each mini-batch is doubled with every sample turned 180 degrees
# (mdp/symmetry.turn_around).
# 256 envs x 128 steps = 32,768 samples per update; 4000 iterations = 131M steps.

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ...mdp.symmetry import turn_around


@configclass
class ExcavatorNavigatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 4000
    save_interval = 25
    experiment_name = "excavator_navigate"
    obs_groups = {"actor": ["policy"], "critic": ["policy", "critic"]}
    clip_actions = 1.0

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
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
        symmetry_cfg=RslRlSymmetryCfg(use_data_augmentation=True, data_augmentation_func=turn_around),
    )
