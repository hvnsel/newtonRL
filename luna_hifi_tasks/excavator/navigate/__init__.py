# Navigate task ids: Luna-Excavator-Navigate (training),
# Luna-Excavator-Navigate-Course (the mdp/course.py obstacle course, playback)
# and Luna-Excavator-Navigate-RFT (the wheels on the RFT soil, for fine-tuning).

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

gym.register(
    id="Luna-Excavator-Navigate-Course",
    entry_point=f"{__name__}.navigate_env:ExcavatorNavigateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigate_env_cfg:ExcavatorNavigateCourseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ExcavatorNavigatePPORunnerCfg",
    },
)

gym.register(
    id="Luna-Excavator-Navigate-RFT",
    entry_point=f"{__name__}.navigate_rft_env:ExcavatorNavigateRftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.navigate_env_cfg:ExcavatorNavigateRftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ExcavatorNavigateRftPPORunnerCfg",
    },
)
