import pytest
from bystro.utils.compress import (
    get_compress_from_pipe_cmd,
    get_decompress_to_pipe_cmd,
    _get_gzip_program_path,
    NUM_CPUS,
)


@pytest.mark.parametrize(
    "which_returns, expected_path",
    [
        ({"bgzip": "/usr/bin/bgzip", "pigz": None, "gzip": None}, "/usr/bin/bgzip"),
        ({"bgzip": None, "pigz": "/usr/bin/pigz", "gzip": "/usr/bin/gzip"}, "/usr/bin/pigz"),
        ({"bgzip": None, "pigz": None, "gzip": "/usr/bin/gzip"}, "/usr/bin/gzip"),
        ({"bgzip": None, "pigz": None, "gzip": None}, None),
    ],
)
def test_get_gzip_program_path(monkeypatch, which_returns, expected_path):
    def mock_shutil_which(program_name):
        return which_returns.get(program_name)

    monkeypatch.setattr("shutil.which", mock_shutil_which)

    if expected_path is None:
        with pytest.raises(OSError):
            _get_gzip_program_path()
    else:
        assert _get_gzip_program_path() == expected_path


@pytest.mark.parametrize(
    "gzip_executable, is_pigz, is_bgzip, is_gzip, out, expected_cmd",
    [
        ("/usr/bin/gzip", False, False, True, "output.gz", "/usr/bin/gzip -c > output.gz"),
        (
            "/usr/bin/pigz",
            True,
            False,
            False,
            "output.gz",
            f"/usr/bin/pigz -c -p {NUM_CPUS} > output.gz",
        ),
        (
            "/usr/bin/bgzip",
            False,
            True,
            False,
            "output.gz",
            f"/usr/bin/bgzip -c --index --index-name output.gz.gzi --threads {NUM_CPUS} > output.gz",
        ),
    ],
)
def test_get_compress_from_pipe_cmd(
    gzip_executable, is_pigz, is_bgzip, is_gzip, out, expected_cmd, monkeypatch
):
    monkeypatch.setattr("bystro.utils.compress.GZIP_EXECUTABLE", gzip_executable)
    monkeypatch.setattr("bystro.utils.compress.IS_PIGZ", is_pigz)
    monkeypatch.setattr("bystro.utils.compress.IS_BGZIP", is_bgzip)
    monkeypatch.setattr("bystro.utils.compress.IS_GZIP", is_gzip)

    if not out.endswith(".gz"):
        with pytest.raises(ValueError):
            get_compress_from_pipe_cmd(out)
    else:
        assert get_compress_from_pipe_cmd(out) == expected_cmd


@pytest.mark.parametrize(
    "executable, is_pigz, is_bgzip, is_gzip, expected_cmd",
    [
        ("/usr/bin/bgzip", False, True, False, f"/usr/bin/bgzip -d -c input.gz --threads {NUM_CPUS}"),
        ("/usr/bin/pigz", True, False, False, "/usr/bin/pigz -d -c input.gz -p 2"),
        ("/usr/bin/gzip", False, False, True, "/usr/bin/gzip -d -c input.gz"),
    ],
)
def test_get_decompress_to_pipe_cmd(monkeypatch, executable, is_pigz, is_bgzip, is_gzip, expected_cmd):
    monkeypatch.setattr("bystro.utils.compress.GZIP_EXECUTABLE", executable)
    monkeypatch.setattr("bystro.utils.compress.IS_PIGZ", is_pigz)
    monkeypatch.setattr("bystro.utils.compress.IS_BGZIP", is_bgzip)
    monkeypatch.setattr("bystro.utils.compress.IS_GZIP", is_gzip)
    assert get_decompress_to_pipe_cmd("input.gz") == expected_cmd
