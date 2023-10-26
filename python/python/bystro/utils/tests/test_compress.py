import pytest
import shutil
from bystro.utils.compress import _get_gzip_program_path


def test_pigz_found(monkeypatch):
    monkeypatch.setattr(
        shutil, "which", lambda x: "/path/to/pigz" if x == "pigz" else None
    )
    assert _get_gzip_program_path() == "/path/to/pigz"


def test_pigz_not_found_gzip_found(monkeypatch):
    monkeypatch.setattr(
        shutil, "which", lambda x: "/path/to/gzip" if x == "gzip" else None
    )
    assert _get_gzip_program_path() == "/path/to/gzip"


def test_neither_found(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(OSError, match="Neither gzip nor pigz not found on system"):
        _get_gzip_program_path()
