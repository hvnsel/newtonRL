# A stand-in for omni.client on PACE. omni.client comes from the
# omniverseclient wheel, which needs glibc 2.35; PACE has 2.34, so
# setup_pace.sh leaves it out. cluster/env.sh puts cluster/stubs on PYTHONPATH.
#
# Isaac Lab's utils/assets.py asks omni.client about every file a USD layer
# references that is not on local disk. Spawning the ground plane does that:
# its material names the MDL module OmniPBR.mdl. The real client answers "not
# found" for such a path, and Isaac Lab keeps the reference as written; this
# stand-in gives the same answer for every path. It downloads nothing, so a
# remote URL also reads as not found.
#
# omni/ has no __init__.py, so omni is a namespace package.

from __future__ import annotations

import enum


class Result(enum.IntEnum):
    OK = 0
    ERROR = 1
    ERROR_NOT_FOUND = 2


class CopyBehavior(enum.IntEnum):
    ERROR_IF_EXISTS = 0
    OVERWRITE = 1


def stat(url: str, *args, **kwargs) -> tuple[Result, None]:
    """(result, entry): not found."""
    return Result.ERROR_NOT_FOUND, None


def copy(src_url: str, dst_url: str, *args, **kwargs) -> Result:
    """Not found; nothing is copied."""
    return Result.ERROR_NOT_FOUND


def read_file(url: str, *args, **kwargs) -> tuple[Result, None, None]:
    """(result, version, content): not found."""
    return Result.ERROR_NOT_FOUND, None, None


def set_s3_configuration(*args, **kwargs) -> Result:
    """Asset region profiles are not supported."""
    return Result.ERROR
