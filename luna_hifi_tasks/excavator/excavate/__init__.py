# Excavation skill: fill both drums from an MPM regolith bed.
#
#   isaaclab train --rl_library rsl_rl --task Luna-Excavator-Excavate --num_envs 16
#
# Luna-Excavator-Excavate-Small is the same task on a 4.4 x 1.8 x 0.15 m bed
# (~9.5k particles/env instead of 32k), for watching the MPM coupling work on
# a laptop:
#
#   isaaclab -p scripts/dig_demo.py
#   isaaclab play  --rl_library rsl_rl --task Luna-Excavator-Excavate --num_envs 1 --checkpoint latest --viz newton
#
# Entry points are strings, so importing this package needs only gymnasium.

import gymnasium as gym

from . import agents

gym.register(
    id="Luna-Excavator-Excavate",
    entry_point=f"{__name__}.excavate_env:ExcavatorExcavateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.excavate_env_cfg:ExcavatorExcavateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ExcavatorExcavatePPORunnerCfg",
    },
)

gym.register(
    id="Luna-Excavator-Excavate-Small",
    entry_point=f"{__name__}.excavate_env:ExcavatorExcavateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.excavate_env_cfg:ExcavatorExcavateSmallEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ExcavatorExcavatePPORunnerCfg",
    },
)
