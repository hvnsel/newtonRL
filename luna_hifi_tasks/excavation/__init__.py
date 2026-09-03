# Registers the excavation task with gymnasium. Once this package is pip
# installed (-e) into the Isaac Lab venv:
#
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task Luna-Rover-Excavation-Direct-v0 --num_envs 1024 --headless
#
#   ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
#       --task Luna-Rover-Excavation-Direct-v0 --num_envs 16 --visualizer newton

import gymnasium as gym

from . import agents

gym.register(
    id="Luna-Rover-Excavation-Direct-v0",
    entry_point=f"{__name__}.excavation_env:RoverExcavationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.excavation_env_cfg:RoverExcavationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoverExcavationPPORunnerCfg",
    },
)
