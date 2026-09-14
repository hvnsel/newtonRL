# Navigation skill: drive to a goal pose over already-worked terrain.
#
#   isaaclab train --rl_library rsl_rl --task Luna-Excavator-Navigate --num_envs 1024
#   isaaclab play  --rl_library rsl_rl --task Luna-Excavator-Navigate --num_envs 4 --checkpoint latest --viz newton
#
# Entry points are strings, so importing this package needs only gymnasium.

import gymnasium as gym

from . import agents

gym.register(
    id="Luna-Excavator-Navigate",
    entry_point=f"{__name__}.navigate_env:ExcavatorNavigateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigate_env_cfg:ExcavatorNavigateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ExcavatorNavigatePPORunnerCfg",
    },
)
