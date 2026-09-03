# Tricycle pipeline-validation task.
#
#   ./isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py ^
#       --task Luna-Tricycle-Direct-v0 --num_envs 64 --headless
#
#   ./isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py ^
#       --task Luna-Tricycle-Direct-v0 --num_envs 4 --visualizer newton
#
# Success criterion: mean episode reward (= metres travelled in 2 s) climbs
# from ~0 toward ~3-4 m within a few hundred iterations.

import gymnasium as gym

from . import agents

gym.register(
    id="Luna-Tricycle-Direct",
    entry_point=f"{__name__}.tricycle_env:TricycleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tricycle_env_cfg:TricycleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TricyclePPORunnerCfg",
    },
)
