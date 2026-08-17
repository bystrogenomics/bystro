"""Build the lightweight Bystro Think-only distribution."""

from __future__ import annotations

import os
from pathlib import Path
import tomllib
from typing import Mapping, cast

from setuptools import setup  # type: ignore[import-untyped]  # no upstream type metadata
from setuptools.command.build_py import (  # type: ignore[import-untyped]
    build_py as _build_py,
)


SETUP_ROOT: Path = Path(__file__).resolve().parent
PYTHON_ROOT: Path = SETUP_ROOT.parent
_LIGHTWEIGHT_PACKAGE_MODULES: Mapping[str, frozenset[str]] = {
    "bystro": frozenset({"__init__"}),
    "bystro.api": frozenset({"__init__", "auth"}),
}


class _ThinkBuildPy(_build_py):
    """Limit the shared top-level package to its initializer and type marker."""

    def find_package_modules(
        self,
        package: str,
        package_dir: str,
    ) -> list[tuple[str, str, str]]:
        modules = super().find_package_modules(package, package_dir)
        allowed_modules = _LIGHTWEIGHT_PACKAGE_MODULES.get(package)
        if allowed_modules is None:
            return modules
        return [module for module in modules if module[1] in allowed_modules]


def _version() -> str:
    with (PYTHON_ROOT / "pyproject.toml").open("rb") as handle:
        configuration = tomllib.load(handle)
    project = cast(Mapping[str, object], configuration["project"])
    version = project["version"]
    if not isinstance(version, str):
        raise TypeError("project.version must be a string")
    return version


os.chdir(SETUP_ROOT)
setup(
    name="bystro-think",
    version=_version(),
    description="Lightweight Python client for the Bystro Think agent API",
    long_description=(PYTHON_ROOT / "THINK_API.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://bystro.io",
    project_urls={
        "Documentation": (
            "https://github.com/bystrogenomics/bystro/blob/main/python/THINK_API.md"
        ),
        "Repository": "https://github.com/bystrogenomics/bystro",
    },
    license="MPL-2.0",
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    install_requires=[
        "msgspec>=0.18.6,<1",
        "python-socketio[client]>=5.11.0,<6",
        "requests>=2.31.0,<3",
    ],
    package_dir={"": "../python"},
    packages=["bystro", "bystro.api", "bystro.think"],
    package_data={"bystro": ["py.typed"]},
    cmdclass={"build_py": _ThinkBuildPy},
    options={"egg_info": {"egg_base": "."}},
)
