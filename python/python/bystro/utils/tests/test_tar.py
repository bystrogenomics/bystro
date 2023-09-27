"""Test functions for utils module."""

import subprocess
from typing import Any
from unittest.mock import Mock

import pytest
from bystro.utils.tar import (
    GNU_TAR_EXECUTABLE_NAME,
    GNU_TAR_LINUX,
    GNU_TAR_MACOSX,
    _get_gnu_tar_executable_name,
)


# These tests check that we correctly infer the name of the GNU tar
# executable regardless of the operating system we find ourselves
# running on, or that we raise an appropriate error if we can't
# determine the correct executable.
def test_GNU_TAR_EXECUTABLE_NAME():
    assert GNU_TAR_EXECUTABLE_NAME in ["/usr/bin/tar", "/opt/homebrew/bin/gtar"]


def _mock_subprocess_run_macosx_gnu_tar_installed(args: list[str], **_kwargs: dict[str, Any]) -> Mock:
    """Simulate subprocess runs on macosx if GNU tar installed."""
    if args == ["/usr/bin/uname", "-a"]:
        mock = Mock()
        mock.stdout = "some uname output string with Darwin in it"
    elif args == ["/opt/homebrew/bin/gtar", "--version"]:
        mock = Mock()
        mock.stdout = "some string with tar in it"
    else:
        err_msg = f"Couldn't interpret args: {args} correctly, this is an error in the test."
        raise AssertionError(err_msg)
    return mock


def test_get_gnu_tar_executable_macosx_gnu_tar_installed(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _mock_subprocess_run_macosx_gnu_tar_installed)
    assert GNU_TAR_MACOSX == _get_gnu_tar_executable_name()


def _mock_subprocess_run_macosx_gnu_tar_not_installed(
    args: list[str], **_kwargs: dict[str, Any]
) -> Mock:
    """Simulate subprocess runs on macosx if GNU tar not installed."""
    if args == ["/usr/bin/uname", "-a"]:
        mock = Mock()
        mock.stdout = "some uname output string with Darwin in it"
    elif args == ["/opt/homebrew/bin/gtar", "--version"]:
        raise FileNotFoundError
    else:
        err_msg = f"Couldn't interpret args: {args} correctly, this is an error in the test."
        raise AssertionError(err_msg)
    return mock


def test_get_gnu_tar_executable_macosx_gnu_tar_not_installed(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _mock_subprocess_run_macosx_gnu_tar_not_installed)
    with pytest.raises(RuntimeError):
        _get_gnu_tar_executable_name()


def _mock_subprocess_run_linux(args: list[str], **_kwargs: dict[str, Any]) -> Mock:
    """Simulate subprocess runs on linux."""
    if args == ["/usr/bin/uname", "-a"]:
        mock = Mock()
        mock.stdout = "some uname output string with Linux in it"
    else:
        err_msg = f"Couldn't interpret args: {args} correctly, this is an error in the test."
        raise AssertionError(err_msg)
    return mock


def test_get_gnu_tar_executable_linux(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _mock_subprocess_run_linux)
    assert GNU_TAR_LINUX == _get_gnu_tar_executable_name()


def _mock_subprocess_run_unknown_os(args: list[str], **_kwargs: dict[str, Any]) -> Mock:
    """Simulate subprocess runs some arbitrary unrecognized OS."""
    if args == ["/usr/bin/uname", "-a"]:
        mock = Mock()
        mock.stdout = "some uname output string from unrecognized OS"
    else:
        err_msg = f"Couldn't interpret args: {args} correctly, this is an error in the test."
        raise AssertionError(err_msg)
    return mock


def test_get_gnu_tar_executable_unknown_os(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _mock_subprocess_run_unknown_os)
    with pytest.raises(OSError):
        _get_gnu_tar_executable_name()
