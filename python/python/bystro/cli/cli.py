import argparse

from bystro.api.auth import signup, login, get_user, UserProfile, CachedAuth

from bystro.api.annotation import get_jobs, create_jobs, query, JobBasicResponse, JOB_TYPE_ROUTE_MAP

from bystro.cli.proteomics_cli import add_proteomics_subparser

from bystro.api.streaming import stream_file

from bystro.cli.ancestry import add_ancestry_subparser

import json

def signup_cli(args: argparse.Namespace, print_result=True) -> CachedAuth:
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

    return signup(
        email=args.email,
        password=args.password,
        name=args.name,
        host=args.host,
        port=args.port,
        print_result=print_result,
    )


def login_cli(args: argparse.Namespace) -> CachedAuth:
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

    return login(
        email=args.email,
        password=args.password,
        host=args.host,
        port=args.port,
        print_result=True,
    )


def get_jobs_cli(args: argparse.Namespace) -> list[JobBasicResponse] | dict:
    """
    Fetches the jobs for the given job type, or a single job if a job id is specified.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.

    Returns
    -------
    dict
        The response from the server.

    """

    return get_jobs(job_type=args.type, job_id=args.id, print_result=True)


def create_jobs_cli(args: argparse.Namespace) -> list[dict]:
    """
    Creates 1+ annotation jobs with the given files and assembly.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.
    print_result : bool, optional
        Whether to print the result of the job creation operation, by default True.

    Returns
    -------
    list[dict]
        The newly created annotation job submissions.
    """
    return create_jobs(
        files=args.files,
        names=args.names,
        assembly=args.assembly,
        combine=args.combine,
        no_index=args.no_index,
        print_result=True,
    )


def get_user_cli(_args: argparse.Namespace) -> UserProfile:
    """
    Fetches the user profile.

    Returns
    -------
    UserProfile
        The user profile
    """

    return get_user(print_result=True)


def query_cli(args: argparse.Namespace) -> None:
    """
    Performs a query search within the specified job with the given arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command. The arguments are:
        job_id : str, required
            The unique identifier of the job to query.
        query : str, required
            The search query string to be used for fetching data.
        size : int, optional
            The number of records to retrieve in the query response.
        from_ : int, optional
            The record offset from which to start retrieval in the query.

    Returns
    -------
    QueryResults
        The queried results
    """

    res =  query(
        job_id=args.job_id,
        query=args.query,
        size=args.size,
        from_=args.from_,
    )

    print(json.dumps(res))


def stream_file_cli(args: argparse.Namespace) -> None:
    """
    Fetch the file from the /api/jobs/streamFile endpoint.

    Parameters
    ----------
    job_id : str
        The ID of the job of the job being fetched.
    output : bool, optional
        Whether to fetch the output file (True) or the input file (False), by default False.
    key_path : str, optional
        The key path for the output file, required if `output` is True.
    out_dir : str, optional
        The directory to write the file to, if not specified, the file is written to stdout.
    """
    stream_file(
        job_id=args.job_id,
        output=args.output,
        key_path=args.key_path,
        out_dir=args.out_dir,
        write_stdout=not args.out_dir
    )


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
    login_parser = subparsers.add_parser("login", help="Authenticate with the Bystro API")
    login_parser.add_argument(
        "--host",
        required=True,
        help="Host of the Bystro API server, e.g. https://bystro-dev.emory.edu",
    )
    login_parser.add_argument(
        "--port",
        type=int,
        help=(
            "Port of the Bystro API server, e.g. 443. "
            "Defaults to the standard port for the given protocol (http: 80, https: 443)"
        ),
    )
    login_parser.add_argument("--email", required=True, help="Email to login with")
    login_parser.add_argument("--password", required=True, help="Password to login with")
    login_parser.set_defaults(func=login_cli)

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
        "--port",
        type=int,
        help=(
            "Port of the Bystro API server, e.g. 443. "
            "Defaults to the standard port for the given protocol (http: 80, https: 443)"
        ),
    )
    signup_parser.set_defaults(func=signup_cli)

    user_parser = subparsers.add_parser("get-user", help="Handle user operations")
    user_parser.set_defaults(func=get_user_cli)

    # Adding the jobs sub-command
    create_jobs_parser = subparsers.add_parser("create-annotation", help="Create an annotation")
    create_jobs_parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        type=str,
        help="Paths to files: .vcf and .snp formats accepted. "
        "Each file is treated as a separate annotation",
    )
    create_jobs_parser.add_argument(
        "--names",
        nargs="+",
        type=str,
        help=(
            "Names for each job, corresponding to the files provided. "
            "Must be the same length as the files list. "
            "Defaults to the basename of the files"
        ),
    )
    create_jobs_parser.add_argument(
        "--assembly",
        type=str,
        required=True,
        help="Genome assembly (e.g., hg19 or hg38 for human genomes)",
    )
    create_jobs_parser.add_argument(
        "--combine",
        default=False,
        action="store_true",
        help="Where to combine/stitch the input .vcf/.snp files into a single annotation",
    )
    create_jobs_parser.add_argument(
        "--no-index",
        default=False,
        action="store_true",
        help="Whether or not to create a natural language search index the annotation",
    )
    create_jobs_parser.set_defaults(func=create_jobs_cli)

    jobs_parser = subparsers.add_parser("get-jobs", help="Fetch one job or a list of jobs")
    jobs_parser.add_argument("--id", type=str, help="Get a specific job by ID")
    jobs_parser.add_argument(
        "--type",
        choices=list(JOB_TYPE_ROUTE_MAP.keys()),
        help="Get a list of jobs of a specific type",
    )
    jobs_parser.set_defaults(func=get_jobs_cli)

    query_parser = subparsers.add_parser(
        "query", help="The OpenSearch query string query, e.g. (cadd: >= 20)"
    )
    query_parser.add_argument(
        "--query",
        required=True,
        help="The OpenSearch query string query, e.g. (cadd: >= 20)",
    )
    query_parser.add_argument("--size", default=10, type=int, help="How many records (default: 10)")
    query_parser.add_argument(
        "--from_",
        default=0,
        type=int,
        help="The first record to return from the matching results. Used for pagination",
    )
    query_parser.add_argument("--job_id", required=True, type=str, help="The job id to query")
    query_parser.set_defaults(func=query_cli)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch a file")
    fetch_parser.add_argument("--job_id", help="ID of the job to fetch the file from")
    fetch_parser.add_argument(
        "--output", action="store_true", help="Fetch from output dir instead of the input dir"
    )
    fetch_parser.add_argument(
        "--key_path", help="Key path for the output and input dir, required if --output is used"
    )
    fetch_parser.add_argument(
        "--out_dir", help="Specify --out_dir to write the file to that directory, otherwise stdout"
    )
    fetch_parser.set_defaults(func=stream_file_cli)

    add_proteomics_subparser(subparsers)
    add_ancestry_subparser(subparsers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
