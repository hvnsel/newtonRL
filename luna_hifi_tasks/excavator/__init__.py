# Bucket-drum excavator.
#
# Asset only, for now. There is deliberately no gym.register() here yet and
# this package is NOT imported from luna_hifi_tasks/__init__.py -- registering
# a task id that has no env behind it would put a broken entry in the gym
# registry and break `isaaclab train --task` discovery for everything else.
#
# When the env lands, this file grows a gym.register() the way
# tricycle/__init__.py has one, and luna_hifi_tasks/__init__.py grows a
# matching `from . import excavator`.
#
# Until then the only thing here is excavator.py, which builds the MJCF that
# the offline USD conversion step consumes.

from . import excavator  # noqa: F401
