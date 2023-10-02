"""Utility functions for safely invoking tar in cross-OS manner."""

import subprocess
from enum import Enum
from subprocess import CalledProcessError
from typing import NoReturn


class OperatingSystem(Enum):
    """Represent the operating system on which the current Python process is running."""

    linux = "linux"
    macosx = "macosx"


# here and throughout, fully qualify executable names to avoid privilege escalation


def _get_executable_name(name: str) -> str:
    try:
        result = subprocess.run(
            ["which", name], capture_output=True, text=True, check=True, shell=False  # noqa: S603, S607
        )
    except CalledProcessError as err:
        err_msg = f"executable `{name}` not found on system, is it installed?"
        raise OSError(err_msg) from err
    return result.stdout.strip()


def assert_never(value: NoReturn) -> NoReturn:
    """Raise error for mypy if code is proven reachable."""
    msg = f"Unhandled value: {value} ({type(value).__name__})"
    raise AssertionError(msg)


def _determine_os() -> OperatingSystem:
    """Determine whether we're running on Linux or MacOSX by inspecting uname output."""
    try:
        result = subprocess.run(
            ["/usr/bin/uname", "-a"],  # noqa: S603
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as file_not_found_error:
        err_msg = (
            "Couldn't run `uname -a` in shell to determine OS: only linux and macosx are supported."
        )
        raise OSError(err_msg) from file_not_found_error
    downcased_uname_output = result.stdout.lower()
    if "darwin" in downcased_uname_output:
        return OperatingSystem.macosx
    if "linux" in downcased_uname_output:
        return OperatingSystem.linux
    err_msg = f"Could not determine OS from `uname -a` output: `{result.stdout}`"
    raise OSError(err_msg)


def _get_gnu_tar_executable_name() -> str:
    """Find the name of the GNU tar executable on user's instance."""
    # Macs use bsdtar by default.  bsdtar is not fully compatible with
    # GNU tar, the implementation we use.  So if we're on a mac, we'll
    # want to use 'gtar' instead of 'tar'.
    operating_system = _determine_os()
    if operating_system is OperatingSystem.linux:
        return _get_executable_name("tar")
    if operating_system is OperatingSystem.macosx:
        return _get_executable_name("gtar")
    return assert_never(operating_system)


GNU_TAR_EXECUTABLE_NAME = _get_gnu_tar_executable_name()
