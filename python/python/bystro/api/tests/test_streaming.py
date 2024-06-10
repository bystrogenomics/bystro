from unittest.mock import patch, MagicMock
import io
import gzip
from contextlib import contextmanager
import sys

from bystro.api.streaming import stream_file, stream_and_decompress_file


@contextmanager
def redirect_binary_stdout(new_target):
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


@patch("bystro.api.streaming.authenticate")
@patch("bystro.api.streaming.requests.get")
def test_stream_file(mock_get, mock_authenticate):
    # Setup
    mock_authenticate.return_value = (
        MagicMock(url="http://test-url.com"),
        {"Authorization": "Bearer token"},
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Disposition": 'attachment; filename="testfile.txt"'}
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_get.return_value = mock_response

    # Test streaming to generator
    stream = stream_file(job_id="1234", chunk_size=5)
    assert stream is not None

    chunks = list(stream)
    assert chunks == [b"chunk1", b"chunk2"]

    captured_output = io.BytesIO()
    with redirect_binary_stdout(captured_output):
        stream_file(job_id="1234", write_stdout=True, chunk_size=5)
        output = captured_output.getvalue()
        assert output == b"chunk1chunk2"


@patch("bystro.api.streaming.authenticate")
@patch("bystro.api.streaming.requests.get")
def test_stream_and_decompress_file(mock_get, mock_authenticate):
    # Setup
    mock_authenticate.return_value = (
        MagicMock(url="http://test-url.com"),
        {"Authorization": "Bearer token"},
    )
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Disposition": 'attachment; filename="testfile.txt"'}
    content = gzip.compress(b"chunk1chunk2")
    mock_response.iter_content.return_value = [content[:10], content[10:]]
    mock_get.return_value = mock_response

    # Test decompression
    stream = stream_and_decompress_file(job_id="1234")

    assert stream is not None
    decompressed_chunks = list(stream)
    assert b"".join(decompressed_chunks) == b"chunk1chunk2"
