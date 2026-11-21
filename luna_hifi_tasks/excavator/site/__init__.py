# Site: one machine digging a trench with the trained excavate and navigate
# policies, the passes planned by luna_hifi_tasks/planner/site.py. Run by
# scripts/site_run.py.

import gymnasium as gym

gym.register(
    id="Luna-Excavator-Site",
    entry_point=f"{__name__}.site_env:ExcavatorSiteEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.site_env:ExcavatorSiteEnvCfg"},
)
