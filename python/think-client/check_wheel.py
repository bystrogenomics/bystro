"""Verify that the lightweight release wheel contains only its public SDK."""

from __future__ import annotations

from email.parser import BytesParser
from pathlib import Path
import sys
import tomllib
from typing import Mapping, cast
from zipfile import ZipFile


PACKAGE_FILES: frozenset[str] = frozenset(
    {
        "bystro/__init__.py",
        "bystro/py.typed",
        "bystro/api/__init__.py",
        "bystro/api/auth.py",
        "bystro/think/__init__.py",
        "bystro/think/client.py",
        "bystro/think/context.py",
        "bystro/think/errors.py",
        "bystro/think/models.py",
        "bystro/think/progress.py",
    }
)
DIST_INFO_FILES: frozenset[str] = frozenset(
    {"METADATA", "RECORD", "WHEEL", "top_level.txt"}
)


def _expected_version() -> str:
    configuration_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with configuration_path.open("rb") as handle:
        configuration = tomllib.load(handle)
    project = cast(Mapping[str, object], configuration["project"])
    version = project["version"]
    if not isinstance(version, str):
        raise TypeError("project.version must be a string")
    return version


def check_wheel(wheel_path: Path) -> None:
    """Reject corrupt wheels, unexpected modules, and mismatched metadata."""

    version = _expected_version()
    dist_info = f"bystro_think-{version}.dist-info"
    expected_files = PACKAGE_FILES | {
        f"{dist_info}/{filename}" for filename in DIST_INFO_FILES
    }
    with ZipFile(wheel_path) as wheel:
        corrupt_file = wheel.testzip()
        if corrupt_file is not None:
            raise ValueError(f"wheel contains a corrupt file: {corrupt_file}")
        actual_files = set(wheel.namelist())
        if actual_files != expected_files:
            unexpected = sorted(actual_files - expected_files)
            missing = sorted(expected_files - actual_files)
            raise ValueError(
                f"unexpected wheel contents (unexpected={unexpected}, missing={missing})"
            )
        metadata = BytesParser().parsebytes(wheel.read(f"{dist_info}/METADATA"))
    if metadata["Name"] != "bystro-think" or metadata["Version"] != version:
        raise ValueError("wheel metadata does not match the lightweight release")
    if metadata["Description-Content-Type"] != "text/markdown":
        raise ValueError("wheel README metadata is not Markdown")


def main(arguments: list[str]) -> None:
    if len(arguments) != 1:
        raise SystemExit("usage: check_wheel.py PATH")
    check_wheel(Path(arguments[0]))


if __name__ == "__main__":
    main(sys.argv[1:])
