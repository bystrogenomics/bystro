"""Utility functions for proteomics module."""

import subprocess
from enum import Enum


class OperatingSystem(Enum):
    """Represent the operating system on which code is currently running."""

    linux = "linux"
    macosx = "macosx"


# here and throughout, fully qualify executable names to avoid privilege escalation
GNU_TAR_LINUX = "/usr/bin/tar"
GNU_TAR_MACOSX = "/opt/homebrew/bin/gtar"


def _determine_os() -> OperatingSystem:
    """Determine whether we're running on Linux or MacOSX by inspecting uname output."""
    try:
        result = subprocess.run(
            ["/usr/bin/uname", "-a"],  # noqa: S603 (this input is safe)
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as file_not_found_error:
        err_msg = "Couldn't run `uname -a` in shell to determine OS: "
        err_msg += "only linux and macosx are supported."
        raise AssertionError(err_msg) from file_not_found_error
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
        return GNU_TAR_LINUX
    if operating_system is OperatingSystem.macosx:
        _assert_gtar_installed_on_macosx()
        return GNU_TAR_MACOSX
    err_msg = (
        "Tried to determine name of GNU tar executable for operating system"
        f"but didn't recognize operating system: `{operating_system}`"
    )
    raise OSError(err_msg)


def _assert_gtar_installed_on_macosx() -> None:
    """Check that gtar is installed, raising RuntimeError if not."""
    try:
        subprocess.run(
            [GNU_TAR_MACOSX, "--version"],  # noqa: S603 (this input is safe)
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as file_not_found_error:
        err_msg = "Could not detect GNU tar on system, try: `brew install gnu-tar`"
        raise RuntimeError(err_msg) from file_not_found_error


GNU_TAR_EXECUTABLE_NAME = _get_gnu_tar_executable_name()
