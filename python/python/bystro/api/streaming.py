import requests
from bystro.api.auth import authenticate

GET_STREAM_ENDPOINT = "/api/jobs/streamingFile/"

def streaming_file(
    job_id: str,
    output: bool = False,
    key_path: str | None = None,
    print_result: bool = True,
) -> None:
    """
    Fetch the file from the /api/jobs/streamingFile endpoint.

    Parameters
    ----------
    job_id : str
        The ID of the job of the job being fetched.
    output : bool, optional
        Whether to fetch the output file (True) or the input file (False), by default False.
    key_path : str, optional
        The key path for the output file, required if `output` is True.
    print_result : bool
        Whether to print the result of the fetch operation, by default True.
    """

    if not job_id:
        raise ValueError("Please specify either a job id")

    state, auth_header = authenticate()
    url = state.url + GET_STREAM_ENDPOINT

    payload = {
        "_id" : job_id,
        "output": output,
    }

    if output:
        if key_path is None:
            raise ValueError(
                "key_path is required when output is True"
            )

        payload["keyPath"] = key_path

    response = requests.post(url, headers=auth_header, json=payload)

    if response.status_code == 200:

        content_disposition = response.headers.get('Content-Disposition')
        filename = content_disposition.split('filename=')[-1].strip("\"'")

        with open(filename, 'wb') as file:
            file.write(response.content)

        if print_result:
            print(f"File was fetched and saved as {filename} successfully.")
    else:
        raise RuntimeError(
            f"Failed to fetch file. Error code: {response.status_code}"
        )