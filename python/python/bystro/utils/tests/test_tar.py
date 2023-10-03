"""Test functions for utils module."""
import shutil
import subprocess
import sys
from subprocess import CalledProcessError
from typing import Any, Callable, NoReturn
from unittest.mock import Mock

import pytest
from bystro.utils.tar import (
    GNU_TAR_EXECUTABLE_NAME,
    _get_gnu_tar_executable_name,
)


def test_GNU_TAR_EXECUTABLE_NAME():
    assert GNU_TAR_EXECUTABLE_NAME in ["/usr/bin/tar", "/opt/homebrew/bin/gtar"]


# These tests check that we correctly infer the name of the GNU tar
# executable regardless of the operating system we find ourselves
# running on, or that we raise an appropriate error if we can't
# determine the correct executable.  The
# _mock_subprocess_run_... methods mocks all interaction with
# subprocess.run under various OS scenarios and the tests ensure that
# we return the correct GNU tar executable name or raise the appropriate
# exception.


def _mock_subprocess(stdout_string: str) -> Callable[[Any, Any], Mock]:
    """Create a subprocess.run mock that returns stdout_string."""
    mock = Mock()
    mock.stdout = stdout_string

    def f(*args: list[str], **kwargs: dict[str, Any]) -> Mock:
        del args, kwargs
        return mock

    return f


def test_get_gnu_tar_executable_macosx_gnu_tar_installed(monkeypatch):
    mock = Mock()
    mock.stdout = "/opt/homebrew/bin/gtar\n"
    monkeypatch.setattr(subprocess, "run", _mock_subprocess("/opt/homebrew/bin/gtar\n"))
    assert "/opt/homebrew/bin/gtar" == _get_gnu_tar_executable_name()


def test_get_gnu_tar_executable_macosx_gnu_tar_not_installed(monkeypatch):
    def raise_error(args: list[str], kwargs: dict[str, Any]) -> NoReturn:
        del args, kwargs
        raise CalledProcessError(cmd=["which", "gtar"], returncode=1)

    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(OSError, match="executable `gtar` not found on system"):
        _get_gnu_tar_executable_name()


def test_get_gnu_tar_executable_linux(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _mock_subprocess("/usr/bin/tar\n"))
    monkeypatch.setattr(sys, "platform", "linux")
    assert "/usr/bin/tar" == _get_gnu_tar_executable_name()


def test_get_gnu_tar_executable_unknown_os(monkeypatch):
    monkeypatch.setattr(sys, "platform", "some unknown OS")
    with pytest.raises(OSError, match="got some unknown OS instead"):
        _get_gnu_tar_executable_name()
