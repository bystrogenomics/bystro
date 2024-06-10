from msgspec import Struct, json as mjson
import datetime
import os
import requests
import sys
import uuid
import concurrent.futures

from bystro.api.auth import authenticate

FILE_UPLOAD_TIMEOUT = int(os.getenv("FILE_UPLOAD_TIMEOUT", f"{60 * 60 * 24 * 7}"))  # 1 week

JOB_TYPE_ROUTE_MAP = {
    "all": "/list/all",
    "public": "/list/all/public",
    "shared": "/list/shared",
    "incomplete": "/list/incomplete",
    "completed": "/list/completed",
    "failed": "/list/failed",
}


class JobBasicResponse(Struct, rename="camel"):
    """
    The basic job information, returned in job list commands

    Attributes
    ----------
    _id : str
        The id of the job.
    name : str
        The name of the job.
    createdAt : str
        The date the job was created.
    """

    _id: str
    name: str
    createdAt: datetime.datetime


def _generate_uuid():
    return str(uuid.uuid4())


def get_jobs(job_type=None, job_id=None, print_result=False) -> list[JobBasicResponse] | dict:
    """
    Fetches the jobs for the given job type, or a single job if a job id is specified.

    Parameters
    ----------
    job_type : str, optional
        The type of jobs to fetch.
    job_id : str, optional
        The ID of a specific job to fetch.
    print_result : bool, optional
        Whether to print the result of the job fetch operation, by default True.

    Returns
    -------
    dict or list[JobBasicResponse]
        The response from the server.
    """
    state, auth_header = authenticate()
    url = state.url + "/api/jobs"

    if not (job_id or job_type):
        raise ValueError("Please specify either a job id or a job type")

    if job_id and job_type:
        raise ValueError("Please specify either a job id or a job type, not both")

    if not job_id and job_type not in JOB_TYPE_ROUTE_MAP.keys():
        raise ValueError(
            f"Invalid job type: {job_type}. Valid types are: {', '.join(JOB_TYPE_ROUTE_MAP.keys())}"
        )

    url = url + f"/{job_id}" if job_id else url + JOB_TYPE_ROUTE_MAP[job_type]

    if print_result:
        if job_id:
            print(f"\nFetching job with id:\t{job_id}")
        else:
            print(f"\nFetching jobs of type:\t{job_type}")

    response = requests.get(url, headers=auth_header, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"Fetching jobs failed with response status: {response.status_code}. Error: {response.text}"
        )

    if print_result:
        print("\nJob(s) fetched successfully: \n")
        print(mjson.format(response.text, indent=4))
        print("\n")

    if job_id:
        return mjson.decode(response.text, type=dict)

    return mjson.decode(response.text, type=list[JobBasicResponse])


def _upload_file(file_path, url: str, headers: dict, payload: dict):
    with open(file_path, "rb") as f:
        file_request = [
            (
                "file",
                (
                    os.path.basename(file_path),
                    f,
                    "application/octet-stream",
                ),
            )
        ]

        response = requests.post(
            url, headers=headers, data=payload, files=file_request, timeout=FILE_UPLOAD_TIMEOUT
        )

    return response


def create_jobs(
    files,
    assembly: str,
    combine=False,
    names: list[str] | None = None,
    no_index=False,
    print_result=False,
) -> list[dict]:
    """
    Creates 1+ annotation jobs

    Parameters
    ----------
    files : list[str]
        List of file paths for job creation.
    assembly : str
        Genome assembly (e.g., hg19, hg38).
    combine : bool, optional
        Whether to combine the input files into a single annotation dataset/job, by default False.
    names : list[str], optional
        List of names for the annotation jobs, one per file. If not provided, the file name will be used.
    no_index : bool, optional
        Whether to skip creation of a search index for the annotation, by default False.
    print_result : bool, optional
        Whether to print the result of the job creation operation, by default True.

    Returns
    -------
    list[dict]
        The annotation submissions
    """
    state, auth_header = authenticate()
    file_upload_url = state.url + "/api/jobs/upload/"
    job_create_url = state.url + "/api/jobs/create"

    combine = len(files) > 1 and combine

    # We cannot have duplicate file names (not just unique paths, but unique names)
    # because the server will not be able to distinguish between them
    file_basenames = [os.path.basename(file) for file in files]
    no_duplicate_files = len(file_basenames) == len(set(file_basenames))

    if not no_duplicate_files:
        raise ValueError("Duplicate file names detected. Please provide unique file names")

    if combine:
        # To create a singel job,
        # 1. Upload all files to a common folder (the uuid folder)
        # 2. Create a job with the uuid as an input
        # 3. Create a merged job
        job_uuid = _generate_uuid()

        if names:
            if len(names) > 1:
                raise ValueError("Cannot specify more than  for combined job")
        else:
            names = [os.path.basename(files[0])]

        job_metadata = {
            "job": {"assembly": assembly, "options": {"index": not no_index}, "name": names[0]},
            "uuid": job_uuid,
            "inputFileNames": [os.path.basename(file) for file in files],
        }

        file_upload_metadata = {"uuid": job_uuid}

        files_uploaded = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {
                executor.submit(
                    _upload_file,
                    file_path=file,
                    url=file_upload_url,
                    payload=file_upload_metadata,
                    headers=auth_header,
                ): file
                for file in files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                response = future.result()
                if response.status_code != 200:
                    raise RuntimeError(
                        (
                            f"File upload failed for {future_to_file[future]} with status: "
                            f"{response.status_code}. Error: \n{response.text}\n"
                        )
                    )
                if print_result:
                    print(f"\nFile upload successful {future_to_file[future]}:\n")
                    print(mjson.format(response.text, indent=4))
                    print("\n")
                files_uploaded.append(response.json())

        if len(files_uploaded) != len(files):
            raise RuntimeError("Failed to upload all files")

        headers = {
            "Content-Type": "application/json",
            "Authorization": auth_header["Authorization"],
        }
        response = requests.put(job_create_url, headers=headers, json=job_metadata, timeout=30)

        if response.status_code != 200:
            raise RuntimeError(
                f"Job creation failed with response status: {response.status_code}.\
                    Error: \n{response.text}\n"
            )

        if print_result:
            print("\nJob creation successful:\n")
            print(mjson.format(response.text, indent=4))
            print("\n")

        return [response.json()]

    if names is None:
        names = [os.path.basename(file) for file in files]
    elif len(names) != len(files):
        raise ValueError("Number of names must match the number of files")

    job_metadata = {}
    for i, file in enumerate(files):
        payload = {
            "job": mjson.encode(
                {"assembly": assembly, "options": {"index": not no_index}, "name": names[i]}
            )
        }

        job_metadata[file] = payload

    jobs_created = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {
            executor.submit(
                _upload_file,
                file_path=file,
                url=file_upload_url,
                payload=job_metadata[file],
                headers=auth_header,
            ): file
            for file in files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            response = future.result()
            if response.status_code != 200:
                raise RuntimeError(
                    (
                        f"File upload and job creation failed for {future_to_file[future]} with status: "
                        f"{response.status_code}. Error: \n{response.text}\n"
                    )
                )
            if print_result:
                print(f"\nJob creation successful for {future_to_file[future]}")
                print(mjson.format(response.text, indent=4))
                print("\n")
            jobs_created.append(response.json())

    return jobs_created


def query(job_id, query, size=10, from_=0):
    """
    Performs a query search within the specified job with the given arguments.

    Parameters
    ----------
    query : str, required
        The search query string to be used for fetching data.
    size : int, optional
        The number of records to retrieve in the query response.
    from_ : int, optional
        The record offset from which to start retrieval in the query.
    job_id : str, required
        The unique identifier of the job to query.

    Returns
    -------
    QueryResults
        The queried results
    """

    state, auth_header = authenticate()

    try:
        query_payload = {
            "from": from_,
            "query": {
                "bool": {
                    "must": {
                        "query_string": {
                            "default_operator": "AND",
                            "query": query,
                            "lenient": True,
                            "phrase_slop": 5,
                            "tie_breaker": 0.3,
                        }
                    }
                }
            },
            "size": size,
        }

        response = requests.post(
            state.url + f"/api/jobs/{job_id}/search",
            headers=auth_header,
            json={"id": job_id, "searchBody": query_payload},
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Query failed with status: {response.status_code}. Error: \n{response.text}\n"
            )

        query_results = response.json()

        return query_results

    except Exception as e:
        sys.stderr.write(f"Query failed: {e}\n")
