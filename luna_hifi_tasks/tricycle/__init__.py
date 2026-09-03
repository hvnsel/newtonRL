# Tricycle pipeline-validation task.
#
# Importing this module registers the task with Gymnasium. Isaac Lab imports
# it through the `isaaclab.tasks` entry point declared in pyproject.toml, so
# the CLI can find it:
#
#   .\isaaclab.bat train --rl_library rsl_rl --task Luna-Tricycle-Direct --num_envs 4
#   .\isaaclab.bat play  --rl_library rsl_rl --task Luna-Tricycle-Direct --num_envs 1 --checkpoint latest --viz newton
#
# Success criterion: mean episode reward (= metres travelled in 2 s) climbs
# from ~0 and keeps climbing without the physics diverging.

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
