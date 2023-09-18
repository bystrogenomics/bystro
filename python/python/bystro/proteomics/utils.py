"""Utility functions for proteomics module."""

from enum import Enum
import subprocess


class OperatingSystem(Enum):
    """Represent the operating system on which code is currently running."""

    linux = "linux"
    macosx = "macosx"


GNU_TAR_LINUX = "tar"
GNU_TAR_MACOSX = "gtar"


def _determine_os() -> OperatingSystem:
    """Determine whether we're running on Linux or MacOSX by inspecting uname output."""
    result = subprocess.run(["uname", "-a"], capture_output=True, text=True)
    if "Darwin" in result.stdout:
        return OperatingSystem.macosx
    elif "Linux" in result.stdout:
        return OperatingSystem.linux
    else:
        err_msg = f"Could not determine OS from `uname -a` output: {result.stdout}"
        raise RuntimeError(err_msg)


def _get_gnu_tar_executable_name() -> str:
    """Find the name of the GNU tar executable on user's instance."""
    # Macs use bsdtar by default.  bsdtar is not fully compatible with
    # GNU tar, the implementation we use.  So if we're on a mac, we'll
    # want to use 'gtar' instead of 'tar'.
    operating_system = _determine_os()
    if operating_system is OperatingSystem.linux:
        return GNU_TAR_LINUX
    elif operating_system is OperatingSystem.macosx:
        _assert_gtar_installed()
        return GNU_TAR_MACOSX
    else:
        err_msg = (
            "Tried to determine name of GNU tar executable for operating system"
            f"but didn't recognize operating system: {operating_system}"
        )
        raise AssertionError(err_msg)


def _assert_gtar_installed() -> None:
    """Check that gtar is installed, raising RuntimeError if not."""
    try:
        result = subprocess.run([GNU_TAR_MACOSX, "--version"], capture_output=True, text=True)
    except FileNotFoundError as fnf_err:
        err_msg = (
            "It looks like you're on a mac but don't have GNU tar installed, "
            "try: `brew install gnu-tar`"
        )
        raise RuntimeError(err_msg) from fnf_err


GNU_TAR_EXECUTABLE_NAME = _get_gnu_tar_executable_name()
