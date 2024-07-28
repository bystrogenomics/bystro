import io
import gzip
import requests
import sys
from pathlib import Path
from typing import Generator
from tqdm import tqdm
from bystro.api.auth import authenticate

GET_STREAM_ENDPOINT = "/api/jobs/{job_id}/streamFile"
MAX_STREAM_TIMEOUT = 60 * 60 * 24 * 7  # 1 week


def stream_file(
    job_id: str,
    output: bool = False,
    key_path: str | None = None,
    out_dir: str | None = None,
    write_stdout: bool = False,
    chunk_size: int | None = 1024 * 1024 * 10,
) -> None | Generator[bytes, None, None]:
    """
    Fetch the file from the /api/jobs/:id/streamFile endpoint.

    Parameters
    ----------
    job_id : str
        The ID of the job being fetched.
    output : bool, optional
        Whether to fetch the output file (True) or the input file (False), by default False.
    key_path : str, optional
        The path to the desired file, required if `output` is True and it will direct to the output dir.
        If `output` is False, `key_path` is used as an index into the input file dir. If not provided,
        the first input file is used.
    out_dir : str, optional
        If specified, write the file to this directory. If not specified, the file is written to stdout
    write_stdout : bool, optional
        If True, write the file to write_stdout. If False, write the file to the specified directory.
    chunk_size : int, optional
        The size of the chunks to read from the response, by default 10MB

    Returns
    -------
    None | Generator[bytes, None, None]
        If `out_dir` is not specified, a generator that yields the file in chunks. Otherwise, None.

    Examples
    --------
    >>> from bystro.api import auth, annotation, streaming
    >>> import gzip
    >>> import io
    >>> user = auth.login('email', 'password', 'https://bystro-dev.emory.edu', print_result=False)
    >>> res = annotation.get_jobs('completed')
    >>> job_id = res[-2]._id
    # Create a buffer to hold the streamed data
    >>> buffer = io.BytesIO()

    # Create a GzipFile object to read the decompressed data
    >>> decompressor = gzip.GzipFile(fileobj=buffer, mode='rb')

    >>> for chunk in streaming.stream_file(job_id, output=True, key_path='annotation'):
    >>>     buffer.write(chunk)
    >>>     buffer.seek(0)  # Reset buffer position to the start
    >>>     while True:
    >>>         try:
    >>>             data = decompressor.read(1024)  # Read decompressed data in chunks
    >>>             if not data:
    >>>                 break
    >>>             print(data.decode('utf-8'))  # Adjust the decoding if needed
    >>>         except EOFError:
    >>>             break
    >>>     # Clear the buffer after reading
    >>>     buffer.seek(0)
    >>>     buffer.truncate()

    # Close the decompressor
    >>> decompressor.close()
    """
    if not job_id:
        raise ValueError("Please specify a job id")

    state, auth_header = authenticate()
    url = state.url + GET_STREAM_ENDPOINT.format(job_id=job_id)

    payload: dict[str, bool | str | int] = {
        "output": output,
    }

    if output:
        if key_path is None:
            raise ValueError("key_path is required when output is True")
        payload["keyPath"] = key_path
    else:
        payload["keyPath"] = key_path if key_path else 0

    response = requests.get(
        url, headers=auth_header, json=payload, stream=True, timeout=MAX_STREAM_TIMEOUT
    )

    if response.status_code == 200:
        content_disposition = response.headers.get("Content-Disposition")
        if not content_disposition:
            raise RuntimeError("No Content-Disposition header found in the response.")

        filename = content_disposition.split("filename=")[-1].strip("\"'")

        total_size = int(response.headers.get("Content-Length", 0))

        if out_dir:
            out_dir_path = Path(out_dir)
            out_dir_path.mkdir(parents=True, exist_ok=True)
            out_file = out_dir_path / filename
            with (
                tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar,
                open(out_file, "wb") as f,
            ):
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            return None

        if write_stdout:
            if hasattr(sys.stdout, "buffer"):
                with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            progress_bar.update(len(chunk))
                            sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            else:
                with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            progress_bar.update(len(chunk))
                            sys.stdout.write(chunk)
                    sys.stdout.flush()

            return None

        def generator():
            with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    progress_bar.update(len(chunk))
                    yield chunk

        return generator()

    if response.status_code == 400:
        raise RuntimeError(f"Bad Request: {response.text}")

    if response.status_code == 404:
        raise RuntimeError("File not found.")

    raise RuntimeError(f"Failed to fetch file. Error code: {response.status_code}")


def stream_and_decompress_file(
    job_id: str,
    output: bool = True,
    key_path: str | None = "annotation",
) -> None | Generator[bytes, None, None]:
    """
    Fetch the file from the /api/jobs/:id/streamFile endpoint and decompress it.

    Parameters
    ----------
    job_id : str
        The ID of the job being fetched.
    output : bool, optional
        Whether to fetch the output file (True) or the input file (False), by default False.
    key_path : str, optional
        The path to the desired file, required if `output` is True and it will direct to the output dir.
        If `output` is False, `key_path` is used as an index into the input file dir. If not provided,
        the first input file is used.

    Returns
    -------
    None | Generator[bytes, None, None]
        If `out_dir` is not specified, a generator that yields the file in chunks. Otherwise, None.

    Examples
    --------
    >>> line = ''
    >>> lines = []
    >>> for chunk in streaming.stream_and_decompress_file(job_id, output=True, key_path='annotation'):
    >>>    line_chunk = chunk.decode('utf-8')

    >>>    if "\n" in line_chunk:
    >>>        parts = line_chunk.split("\n")
    >>>        lines.append(line + parts[0])
    >>>        line = parts[1]

    """

    # Create a buffer to hold the streamed data
    buffer = io.BytesIO()

    # Create a GzipFile object to read the decompressed data
    decompressor = gzip.GzipFile(fileobj=buffer, mode="rb")

    stream = stream_file(job_id, output=output, key_path=key_path)

    if stream is None:
        return None

    def generator():
        for chunk in stream:
            buffer.write(chunk)
            buffer.seek(0)  # Reset buffer position to the start
            while True:
                try:
                    data = decompressor.read(1024)
                    if not data:
                        break
                    yield data
                except EOFError:
                    break

            # Clear the buffer after reading
            buffer.seek(0)
            buffer.truncate()

        decompressor.close()

    return generator()
