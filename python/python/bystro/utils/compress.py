"""Utility functions for safely invoking tar in cross-OS manner."""
import shutil


def _get_gzip_program_path() -> str:
    pigz = shutil.which("pigz")

    if pigz:
        return pigz

    gzip = shutil.which("gzip")

    if gzip:
        return gzip

    raise OSError("Neither gzip nor pigz not found on system")


GZIP_EXECUTABLE = _get_gzip_program_path()
