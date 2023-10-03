"""Utility functions for safely invoking tar in cross-OS manner."""
import sys
import shutil
import subprocess
from enum import Enum
from subprocess import CalledProcessError
from typing import NoReturn


# here and throughout, fully qualify executable names to avoid privilege escalation


def _get_executable_name(name: str) -> str:
    executable_name = shutil.which(name)
    if executable_name is None:
        err_msg = f"executable `{name}` not found on system"
        raise OSError(err_msg)
    return executable_name


def assert_never(value: NoReturn) -> NoReturn:
    """Raise error for mypy if code is proven reachable."""
    msg = f"Unhandled value: {value} ({type(value).__name__})"
    raise AssertionError(msg)


def _get_gnu_tar_executable_name() -> str:
    """Find the name of the GNU tar executable on user's instance."""
    # Macs use bsdtar by default.  bsdtar is not fully compatible with
    # GNU tar, the implementation we use.  So if we're on a mac, we'll
    # want to use 'gtar' instead of 'tar'.
    if sys.platform == "linux":
        return _get_executable_name("tar")
    if sys.platform == "darwin":
        return _get_executable_name("gtar")
    err_msg = f"Operating system must be linux or macosx, got {sys.platform} instead."
    raise OSError(err_msg)


GNU_TAR_EXECUTABLE_NAME = _get_gnu_tar_executable_name()
