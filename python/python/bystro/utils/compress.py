"""Utility functions for safely invoking tar in cross-OS manner."""

import shutil
import multiprocessing

NUM_CPUS = multiprocessing.cpu_count()


def _get_gzip_program_path() -> str:
    bgzip = shutil.which("bgzip")

    if bgzip:
        return bgzip

    pigz = shutil.which("pigz")

    if pigz:
        return pigz

    gzip = shutil.which("gzip")

    if gzip:
        return gzip

    raise OSError("None of bgzip, pigz, gzip found on system")


GZIP_EXECUTABLE = _get_gzip_program_path()
IS_PIGZ = GZIP_EXECUTABLE.endswith("pigz")
IS_GZIP = GZIP_EXECUTABLE.endswith("gzip")
IS_BGZIP = GZIP_EXECUTABLE.endswith("bgzip")


def get_compress_from_pipe_cmd(out: str, pipe_from_cmd: str = None, pipe_to_cmd: str = None) -> str:
    cmd = None
    if (IS_PIGZ or IS_BGZIP or IS_GZIP) and not out.endswith(".gz"):
        raise ValueError("Output path must end with .gz")

    if IS_PIGZ:
        cmd = f"{GZIP_EXECUTABLE} -c -p {NUM_CPUS}"
    elif IS_BGZIP:
        cmd = f"{GZIP_EXECUTABLE} -c --index --index-name {out}.gzi --threads {NUM_CPUS}"
    else:
        cmd = f"{GZIP_EXECUTABLE} -c"

    if pipe_from_cmd:
        cmd = f"{pipe_from_cmd} | {cmd}"

    if pipe_to_cmd:
        cmd = f"{cmd} | {pipe_to_cmd}"

    return f"{cmd} > {out}"


def get_decompress_to_pipe_cmd(input_path: str, pipe_to_cmd: str = None) -> str:
    cmd = None
    if IS_PIGZ:
        # pigz doesn't benefit much from more than 2 threads during decompression
        cmd = f"{GZIP_EXECUTABLE} -d -c {input_path} -p 2"
    elif IS_BGZIP:
        cmd = f"{GZIP_EXECUTABLE} -d -c {input_path} --threads {NUM_CPUS}"
    else:
        cmd = f"{GZIP_EXECUTABLE} -d -c {input_path}"

    if pipe_to_cmd:
        cmd = f"{cmd} | {pipe_to_cmd}"

    return cmd
