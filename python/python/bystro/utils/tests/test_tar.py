"""Test functions for utils module."""

import subprocess
from subprocess import CalledProcessError
from typing import Any
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


def _mock_subprocess(response_table: dict[tuple[str], (str | Exception)]) -> Mock:
    """Mock subprocess.run, given dictionary of inputs and outputs."""

    def generate_subprocess_mock(args: list[str], **_kwargs: dict[str, Any]):
        response = response_table.get(tuple(args))
        if response is None:
            err_msg = f"Couldn't find args: {args} in response table, this is an error in the test."
            raise AssertionError(err_msg)
        elif isinstance(response, Exception):
            raise response
        mocked_subprocess_response = Mock()
        mocked_subprocess_response.stdout = response
        return mocked_subprocess_response

    return generate_subprocess_mock


def test_get_gnu_tar_executable_macosx_gnu_tar_installed(monkeypatch):
    subprocess_response_table = {
        ("/usr/bin/uname", "-a"): "some uname output string with Darwin in it",
        ("which", "gtar"): "/opt/homebrew/bin/gtar\n",
    }
    monkeypatch.setattr(subprocess, "run", _mock_subprocess(subprocess_response_table))
    assert "/opt/homebrew/bin/gtar" == _get_gnu_tar_executable_name()


def test_get_gnu_tar_executable_macosx_gnu_tar_not_installed(monkeypatch):
    subprocess_responses = {
        ("/usr/bin/uname", "-a"): "some string with Darwin in it",
        ("which", "gtar"): CalledProcessError(cmd=["which", "gtar"], returncode=1),
    }

    monkeypatch.setattr(subprocess, "run", _mock_subprocess(subprocess_responses))
    with pytest.raises(OSError, match="executable `gtar` not found on system"):
        response = _get_gnu_tar_executable_name()


def test_get_gnu_tar_executable_linux(monkeypatch):
    subprocess_response_table = {
        ("/usr/bin/uname", "-a"): "some uname output string with Linux in it",
        ("which", "tar"): "/usr/bin/tar\n",
    }
    monkeypatch.setattr(subprocess, "run", _mock_subprocess(subprocess_response_table))
    assert "/usr/bin/tar" == _get_gnu_tar_executable_name()


def test_get_gnu_tar_executable_unknown_os(monkeypatch):
    subprocess_response_table = {
        ("/usr/bin/uname", "-a"): "some uname output string from unknown OS",
    }
    monkeypatch.setattr(subprocess, "run", _mock_subprocess(subprocess_response_table))
    with pytest.raises(OSError, match="Could not determine OS"):
        _get_gnu_tar_executable_name()
