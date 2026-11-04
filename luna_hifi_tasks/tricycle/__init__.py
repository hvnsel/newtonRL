# Tricycle pipeline-validation task: a tricycle drives on MPM soil for 2 s.
# Episode reward is metres travelled forward.
#
# Importing this module registers the task with Gymnasium. Isaac Lab imports it
# through the `isaaclab.tasks` entry point in pyproject.toml.
#
#   .\isaaclab.bat train --rl_library rsl_rl --task Luna-Tricycle-Direct --num_envs 4
#   .\isaaclab.bat play  --rl_library rsl_rl --task Luna-Tricycle-Direct --num_envs 1 --checkpoint latest --viz newton

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
