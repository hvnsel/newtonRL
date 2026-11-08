# Excavation: one straight pass cutting a strip of an MPM regolith bed into
# both drums. The env lowers and spins the tool; the policy drives.
#
#   isaaclab train --rl_library rsl_rl --task Luna-Excavator-Excavate --num_envs 4
#   isaaclab play  --rl_library rsl_rl --task Luna-Excavator-Excavate --num_envs 1 --checkpoint latest --viz newton
#   isaaclab -p scripts/dig_demo.py
#
# -Small is the same task on a 0.9 x 1.1 x 0.16 m pile in front of the machine,
# -Micro on a 1.0 x 1.0 x 0.21 m pile.

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

gym.register(
    id="Luna-Excavator-Excavate-Micro",
    entry_point=f"{__name__}.excavate_env:ExcavatorExcavateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.excavate_env_cfg:ExcavatorExcavateMicroEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ExcavatorExcavatePPORunnerCfg",
    },
)
