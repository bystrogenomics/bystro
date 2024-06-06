from unittest.mock import patch, MagicMock
import io
import gzip
from contextlib import redirect_stdout

from bystro.api.streaming import stream_file, stream_and_decompress_file


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

    with io.BytesIO() as buf, redirect_stdout(buf):
        stream_file(job_id="1234", write_stdout=True, chunk_size=5)
        buf.seek(0)
        assert buf.read() == b"chunk1chunk2"


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
    decompressed_chunks = list(stream)
    assert b"".join(decompressed_chunks) == b"chunk1chunk2"
