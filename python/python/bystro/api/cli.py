import argparse
import datetime
import os
import sys

import requests
import json

from msgspec import Struct, json as mjson

DEFAULT_DIR = os.path.expanduser("~/.bystro")
STATE_FILE = "bystro_authentication_token.json"
JOB_TYPE_ROUTE_MAP = {
    "all": "/list/all",
    "public": "/list/all/public",
    "shared": "/list/shared",
    "incomplete": "/list/incomplete",
    "completed": "/list/completed",
    "failed": "/list/failed",
}


class SignupResponse(Struct, rename="camel"):
    """
    The response body for signing up for Bystro.

    Attributes
    ----------
    access_token : str
        The access token, which authorizes further API requests
    """

    access_token: str


class LoginResponse(Struct, rename="camel"):
    """
    The response body for logging in to Bystro.

    Attributes
    ----------
    access_token : str
        The access token, which authorizes further API requests
    """

    access_token: str


class CachedAuth(Struct, rename="camel"):
    """
    The authentication state.

    Attributes
    ----------
    email : str
        The email of the user.
    access_token : str
        The access token, which authorizes further API requests
    url : str
        The url of the Bystro server.
    """

    email: str
    access_token: str
    url: str


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


class UserProfile(Struct, rename="camel"):
    """
    The response body for fetching the user profile.

    Attributes
    ----------
    options : dict
        The user options.
    _id : str
        The id of the user.
    name : str
        The name of the user.
    email : str
        The email of the user.
    accounts : list[str]
        The accounts of the user.
    role : str
        The role of the user.
    lastLogin : str
        The date the user last logged in.
    """

    _id: str
    options: dict
    name: str
    email: str
    accounts: list[str]
    role: str
    lastLogin: datetime.datetime


def _fq_host(args: argparse.Namespace) -> str:
    """
    Returns the fully qualified host, e.g. https://bystro-dev.emory.edu:443

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.

    Returns
    -------
    str
        The fully qualified host.
    """
    return f"{args.host}:{args.port}"


def load_state(state_dir: str) -> CachedAuth | None:
    """
    Loads the authentication state from the state directory.

    Parameters
    ----------
    state_dir : str
        The directory where the authentication state is saved.

    Returns
    -------
    CachedAuth | None
        The authentication state, or None if the state file doesn't exist.
    """
    path = os.path.join(state_dir, STATE_FILE)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return mjson.decode(f.read(), type=CachedAuth)

    return None


def save_state(data: CachedAuth, state_dir: str, print_result=True) -> None:
    """
    Saves the authentication state to a file.

    Parameters
    ----------
    data : CachedAuth
        The data to save.
    state_dir : str
        The directory where the authentication state will be saved.
    print_result : bool, optional
        Whether to print the result of the save operation, by default True.

    Returns
    --------
    None
    """
    if not os.path.exists(state_dir):
        os.makedirs(state_dir, exist_ok=True)

    save_path = os.path.join(state_dir, STATE_FILE)
    encoded_data = mjson.encode(data).decode("utf-8")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(encoded_data)

    if print_result:
        print(
            f"\nSaved auth credentials to {save_path}:\n{mjson.format(encoded_data, indent=4)}"
        )


def signup(args: argparse.Namespace, print_result=True) -> CachedAuth:
    """
    Signs up for Bystro with the given email, name, and password. Additionally, logs in and
    saves the credentials, to enable API calls without re-authenticating.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.
    print_result : bool, optional
        Whether to print the result of the signup operation, by default True.

    Returns
    -------
    CachedAuth
        The cached authentication state.
    """
    if print_result:
        print(f"\nSigning up for Bystro with email: {args.email}, name: {args.name}")

    fq_host = _fq_host(args)
    url = f"{fq_host}/api/user"

    data = {"email": args.email, "name": args.name, "password": args.password}

    response = requests.put(url, data=data, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Login failed with response status: {response.status_code}. Error: \n{response.text}\n"
        )

    res = mjson.decode(response.text, type=SignupResponse)
    state = CachedAuth(
        access_token=res.access_token,
        url=fq_host,
        email=args.email,
    )

    save_state(
        state,
        args.dir,
        print_result,
    )

    if print_result:
        print("\nSignup & authentication successful. You may now use the Bystro API!\n")

    return state


def login(args: argparse.Namespace, print_result=True) -> CachedAuth:
    """
    Logs in to the server and saves the authentication state to a file.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.
    print_result : bool, optional
        Whether to print the result of the login operation, by default True.

    Returns
    -------
    CachedAuth
        The cached authentication state.
    """
    fq_host = _fq_host(args)

    if print_result:
        print(f"\nLogging into {fq_host} with email: {args.email}.")

    url = f"{fq_host}/api/user/auth/local"

    body = {"email": args.email, "password": args.password}

    response = requests.post(url, data=body, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Login failed with response status: {response.status_code}. Error: \n{response.text}\n"
        )

    res = mjson.decode(response.text, type=LoginResponse)
    state = CachedAuth(access_token=res.access_token, url=fq_host, email=args.email)
    save_state(state, args.dir, print_result)

    if print_result:
        print("\nLogin successful. You may now use the Bystro API!\n")

    return state


def authenticate(args) -> tuple[CachedAuth, dict]:
    """
    Authenticates the user and returns the url, auth header, and email.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.

    Returns
    -------
    tuple[CachedAuth, dict]
        The cached auth credentials and auth header
    """
    state = load_state(args.dir)

    if not state:
        raise ValueError("\n\nYou are not logged in. Please login first.\n")

    header = {"Authorization": f"Bearer {state.access_token}"}
    return state, header


def get_jobs(
    args: argparse.Namespace, print_result=True
) -> list[JobBasicResponse] | dict:
    """
    Fetches the jobs for the given job type, or a single job if a job id is specified.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.
    print_result : bool, optional
        Whether to print the result of the job fetch operation, by default True.

    Returns
    -------
    dict
        The response from the server.

    """
    state, auth_header = authenticate(args)
    url = state.url + "/api/jobs"
    job_type = args.type
    job_id = args.id

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
        job = mjson.decode(response.text, type=dict)
        # MongoDB doesn't support '.' in field names,
        # so we neede to convert the config to string before saving
        # so we decode the nested json here
        job["config"] = mjson.decode(job["config"])
        return job

    return mjson.decode(response.text, type=list[JobBasicResponse])


def create_job(args: argparse.Namespace, print_result=True) -> dict:
    """
    Creates a job for the given files.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.
    print_result : bool, optional
        Whether to print the result of the job creation operation, by default True.

    Returns
    -------
    dict
        The newly created job.
    """
    state, auth_header = authenticate(args)
    url = state.url + "/api/jobs/upload/"

    payload = {
        "job": mjson.encode(
            {
                "assembly": args.assembly,
                "options": {"index": args.index},
            }
        )
    }

    files = []
    for file in args.files:
        files.append(
            (
                "file",
                (
                    os.path.basename(file),
                    open(file, "rb"),  # noqa: SIM115
                    "application/octet-stream",
                ),
            )
        )

    if print_result:
        print(f"\nCreating jobs for files: {','.join(map(lambda x: x[1][0], files))}\n")

    response = requests.post(
        url, headers=auth_header, data=payload, files=files, timeout=30
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

    return response.json()


def get_user(args: argparse.Namespace, print_result=True) -> UserProfile:
    """
    Fetches the user profile.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.
    print_result : bool, optional
        Whether to print the result of the user profile fetch operation, by default True.

    Returns
    -------
    UserProfile
        The user profile
    """
    if print_result:
        print("\n\nFetching user profile\n")

    state, auth_header = authenticate(args)

    response = requests.get(state.url + "/api/user/me", headers=auth_header, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Fetching profile failed with response status: {response.status_code}.\
                Error: \n{response.text}\n"
        )

    user_profile = mjson.decode(response.text, type=UserProfile)

    if print_result:
        print(f"\nFetched Profile for email {state.email}\n")
        print(mjson.format(response.text, indent=4))
        print("\n")

    return user_profile


def query(args: argparse.Namespace) -> None:
    """
    Performs a query search within the specified job with the given arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.
    dir : str, optional
        The directory where the Bystro API login state is saved.
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

    state, auth_header = authenticate(args)

    try:
        query_payload = {
            "from": args.from_,
            "query": {
                "bool": {
                    "must": {
                        "query_string": {
                            "default_operator": "AND",
                            "query": args.query,
                            "lenient": True,
                            "phrase_slop": 5,
                            "tie_breaker": 0.3,
                        }
                    }
                }
            },
            "size": args.size,
        }

        response = requests.post(
            state.url + f"/api/jobs/{args.job_id}/search",
            headers=auth_header,
            json={"id": args.job_id, "searchBody": query_payload},
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(
                (f"Query failed with status: {response.status_code}. "
                f"Error: \n{response.text}\n")
            )

        query_results = response.json()

        print("\nQuery Results:")
        print(json.dumps(query_results, indent=4))

    except Exception as e:
        sys.stderr.write(f"Query failed: {e}\n")


def main():
    """
    The main function for the CLI tool.

    Returns
    -------
    None

    """
    parser = argparse.ArgumentParser(
        prog="bystro-api", description="Bystro CLI tool for making API calls."
    )
    subparsers = parser.add_subparsers(title="commands")

    # Adding the user sub-command
    login_parser = subparsers.add_parser(
        "login", help="Authenticate with the Bystro API"
    )
    login_parser.add_argument(
        "--host",
        required=True,
        help="Host of the Bystro API server, e.g. https://bystro-dev.emory.edu",
    )
    login_parser.add_argument(
        "--port", type=int, default=443, help="Port of the Bystro API server, e.g. 443"
    )
    login_parser.add_argument("--email", required=True, help="Email to login with")
    login_parser.add_argument(
        "--password", required=True, help="Password to login with"
    )
    login_parser.add_argument(
        "--dir", default=DEFAULT_DIR, help="Where to save Bystro API login state"
    )
    login_parser.set_defaults(func=login)

    signup_parser = subparsers.add_parser("signup", help="Sign up to Bystro")
    signup_parser.add_argument(
        "--email",
        required=True,
        help="Email. This will serve as your unique username for login",
    )
    signup_parser.add_argument("--password", required=True, help="Password")
    signup_parser.add_argument(
        "--name",
        required=True,
        help="The name you'd like to use on the Bystro platform",
    )
    signup_parser.add_argument(
        "--host",
        required=True,
        help="Host of the Bystro API server, e.g. https://bystro-dev.emory.edu",
    )
    signup_parser.add_argument(
        "--port", type=int, default=443, help="Port of the Bystro API server, e.g. 443"
    )
    signup_parser.add_argument(
        "--dir", default=DEFAULT_DIR, help="Where to save Bystro API login state"
    )
    signup_parser.set_defaults(func=signup)

    user_parser = subparsers.add_parser("get-user", help="Handle user operations")
    user_parser.add_argument("--profile", action="store_true", help="Get user profile")
    user_parser.add_argument(
        "--dir", default=DEFAULT_DIR, help="Where Bystro API login state is saved"
    )
    user_parser.set_defaults(func=get_user)

    # Adding the jobs sub-command
    create_jobs_parser = subparsers.add_parser("create-job", help="Create a job")
    create_jobs_parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        type=str,
        help="Paths to files: .vcf and .snp formats accepted.",
    )
    create_jobs_parser.add_argument(
        "--assembly",
        type=str,
        required=True,
        help="Genome assembly (e.g., hg19 or hg38 for human genomes)",
    )
    create_jobs_parser.add_argument(
        "--create-index",
        type=bool,
        default=True,
        help="Whether or not to create a natural language search index the annotation",
    )
    create_jobs_parser.add_argument(
        "--dir", default=DEFAULT_DIR, help="Where Bystro API login state is saved"
    )
    create_jobs_parser.set_defaults(func=create_job)

    jobs_parser = subparsers.add_parser(
        "get-jobs", help="Fetch one job or a list of jobs"
    )
    jobs_parser.add_argument("--id", type=str, help="Get a specific job by ID")
    jobs_parser.add_argument(
        "--type",
        choices=list(JOB_TYPE_ROUTE_MAP.keys()),
        help="Get a list of jobs of a specific type",
    )
    jobs_parser.add_argument(
        "--dir", default=DEFAULT_DIR, help="Where Bystro API login state is saved"
    )
    jobs_parser.set_defaults(func=get_jobs)

    query_parser = subparsers.add_parser(
        "query", help="The OpenSearch query string query, e.g. (cadd: >= 20)"
    )
    query_parser.add_argument(
        "--dir", default=DEFAULT_DIR, help="Where Bystro API login state is saved"
    )
    query_parser.add_argument(
        "--query",
        required=True,
        help="The OpenSearch query string query, e.g. (cadd: >= 20)",
    )
    query_parser.add_argument(
        "--size", default=10, type=int, help="How many records (default: 10)"
    )
    query_parser.add_argument(
        "--from_",
        default=0,
        type=int,
        help="The first record to return from the matching results. Used for pagination",
    )
    query_parser.add_argument(
        "--job_id", required=True, type=str, help="The job id to query"
    )
    query_parser.set_defaults(func=query)

    args = parser.parse_args()
    if hasattr(args, "func"):
        try:
            args.func(args)
        except Exception as e:
            print(f"\nSomething went wrong:\t{e}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
