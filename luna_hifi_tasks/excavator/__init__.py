# Bucket-drum excavator tasks.
#
# Importing this package registers two Gymnasium ids:
#
#   Luna-Excavator-Navigate   rigid procedural terrain, thousands of envs
#   Luna-Excavator-Excavate   MPM regolith bed, tens of envs
#
# Registration is string-based (entry points are module:Class strings), so
# importing this package pulls in gymnasium and nothing else. The env and cfg
# modules -- and with them isaaclab and newton -- load only when gym.make is
# called. That keeps `luna_hifi_tasks.excavator.mdp` importable and testable
# on a machine with no simulator, which is where most of its tests run.

from . import excavator  # noqa: F401  (the asset generator)
from . import excavate, navigate  # noqa: F401  (gym.register side effects)
