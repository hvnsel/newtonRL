# agents/rsl_rl_ppo_cfg.py
#
# Model configs in rsl_rl 5's form, which Isaac Lab passes through to the
# runner (isaaclab_rl.rsl_rl.utils).
#
#   * the actor is rsl_rl's CNNModel. The 25-wide "policy" group --
#     velocities, attitude, the goal, the last action -- is normalised and
#     goes straight to the MLP. The two terrain windows arrive as images and
#     each goes through its own CNN encoder, trained by the PPO loss with
#     everything else; no separate autoencoder. Three 3x3 stride-2
#     convolutions take each window down to a 3 x 3 grid of 4 features, 36
#     numbers: far, 27 x 27 at 0.45 m, goes 27 -> 13 -> 6 -> 3, an output
#     cell every 3.6 m; near, 25 x 25 at 0.20 m, goes 25 -> 12 -> 6 -> 3, one
#     every 1.6 m. rsl_rl's "zeros" padding pads a layer only where the
#     stride needs it, so an odd window loses its edge row rather than
#     growing one. The grid still says on which side of the machine a hazard
#     is, which global pooling would average away. The MLP reads 97 inputs,
#     72 of them terrain.
#   * the critic is an MLP over the policy group and the privileged "critic"
#     group, which carries the clean scan, true slip and exact fill: the
#     asymmetric actor-critic. It is discarded after training, so it may read
#     what no rover could.
#   * symmetry: every mini-batch is doubled with each sample turned 180
#     degrees, mdp/symmetry.turn_around, so neither end of the machine is the
#     front. The extra samples cost a second forward and backward pass per
#     mini-batch, not simulation.
#
# gamma 0.997 is an effective horizon of 333 steps, 13 s at 25 Hz, against a
# 750-step episode covering an 18 m traverse. A goal reached 400 steps out
# discounts to 0.30, so the terminal bonus stays visible from the start of a
# long run.

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ...mdp.symmetry import turn_around

# The layer settings both encoders share.
_ENCODER = dict(output_channels=[16, 16, 4], kernel_size=3, stride=2, padding="zeros", activation="elu")


@configclass
class ExcavatorNavigatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 100
    experiment_name = "excavator_navigate"
    obs_groups = {"actor": ["policy", "far_scan", "near_scan"], "critic": ["policy", "critic"]}
    clip_actions = 1.0

    actor = RslRlCNNModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
        cnn_cfg={
            "far_scan": RslRlCNNModelCfg.CNNCfg(**_ENCODER),
            "near_scan": RslRlCNNModelCfg.CNNCfg(**_ENCODER),
        },
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
        # 0.002 let the std run down to 0.04 by iteration 1800, which is a
        # deterministic policy: once there it cannot explore its way out of
        # whatever it has converged to, and it converged to creeping 2.6 m
        # and stopping. Back to 0.005.
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
