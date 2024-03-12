from msgspec import Struct, json as mjson
import datetime
import os
import requests
import sys


from bystro.api.auth import authenticate

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


def get_jobs(job_type=None, job_id=None, print_result=True) -> list[JobBasicResponse] | dict:
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


def create_jobs(
    files, assembly: str, names: list[str] | None = None, index=True, print_result=True
) -> list[dict]:
    """
    Creates 1+ annotation jobs

    Parameters
    ----------
    files : list[str]
        List of file paths for job creation.
    assembly : str
        Genome assembly (e.g., hg19, hg38).
    names : list[str], optional
        List of names for the annotation jobs, one per file. If not provided, the file name will be used.
    index : bool, optional
        Whether to create a search index for the annotation, by default True.
    print_result : bool, optional
        Whether to print the result of the job creation operation, by default True.

    Returns
    -------
    list[dict]
        The annotation submissions
    """
    if names is not None and len(names) != len(files):
        raise ValueError("The number of names must match the number of files")

    state, auth_header = authenticate()
    url = state.url + "/api/jobs/upload/"

    jobs_created = []
    for i, file in enumerate(files):
        name = names[i] if names is not None else os.path.basename(file)
        payload = {
            "job": mjson.encode({"assembly": assembly, "options": {"index": index}, "name": name})
        }

        with open(file, "rb") as f:
            file_request = [
                (
                    "file",
                    (
                        os.path.basename(file),
                        f,
                        "application/octet-stream",
                    ),
                )
            ]

            response = requests.post(
                url, headers=auth_header, data=payload, files=file_request, timeout=30
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Job creation failed with response status: {response.status_code}.\
                    Error: \n{response.text}\n"
            )

        if print_result:
            print("\nJob creation successful:\n")
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

        print("\nQuery Results:")
        print(mjson.format(mjson.encode(query_results), indent=4))

    except Exception as e:
        sys.stderr.write(f"Query failed: {e}\n")
