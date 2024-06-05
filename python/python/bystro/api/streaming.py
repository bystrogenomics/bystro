import requests
import sys
import subprocess
from pathlib import Path

from bystro.api.auth import authenticate

GET_STREAM_ENDPOINT = "/api/jobs/{job_id}/streamFile"

def stream_file(
        job_id: str, output: bool = False, key_path: str | None = None,
        out_dir: str | None = None
    ) -> None:
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
    """
    if not job_id:
        raise ValueError("Please specify a job id")

    state, auth_header = authenticate()
    url = state.url + GET_STREAM_ENDPOINT.format(job_id=job_id)

    payload : dict[str, bool | str | int]  = {
        "output": output,
    }

    if output:
        if key_path is None:
            raise ValueError("key_path is required when output is True")
        payload["keyPath"] = key_path
    else:
        payload["keyPath"] = key_path if key_path else 0

    response = requests.get(url, headers=auth_header, json=payload, stream=True)

    if response.status_code == 200:
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            filename = content_disposition.split("filename=")[-1].strip("\"'")

            # If the output directory is specified, stream the file to that directory
            if out_dir:
                out_dir = Path(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / filename
                with open(out_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        f.write(chunk)
            else:
                # Otherwise, stream the file to stdout
                for chunk in response.iter_content(chunk_size=1024):
                    sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
        else:
            raise RuntimeError("No Content-Disposition header found in the response.")
    elif response.status_code == 400:
        raise RuntimeError(f"Bad Request: {response.text}")
    elif response.status_code == 404:
        raise RuntimeError("File not found.")
    else:
        raise RuntimeError(f"Failed to fetch file. Error code: {response.status_code}")