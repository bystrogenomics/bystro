import argparse
import json
import os

import requests

STATE_FILE = "bystro_authentication_token.json"
TOKEN_KEY = "access_token"
JOB_TYPE_ROUTE_MAP = {
    "all": "/list/all",
    "public": "/list/all/public",
    "shared": "/list/shared",
    "incomplete": "/list/incomplete",
    "completed": "/list/completed",
    "failed": "/list/failed",
}


def load_state(state_dir: str):
    path = os.path.join(state_dir, STATE_FILE)

    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    return {}


def save_state(data: object, state_dir: str, print_result=True):
    save_path = os.path.join(state_dir, STATE_FILE)

    with open(save_path, "w") as f:
        json.dump(data, f)

    if print_result:
        print(
            f"\nSaved authentication credentials to {save_path}:\n{json.dumps(data, indent=4)}"
        )


def signup(args: argparse.Namespace, print_result=True):
    if print_result:
        print(f"\nSigning up for Bystro with email: {args.email}, name: {args.name}")

    url = f"{args.host}:{args.port}/api/user"

    data = {"email": args.email, "name": args.name, "password": args.password}

    response = requests.put(url, data=data)

    if response.status_code != 200:
        raise RuntimeError(
            f"Login failed with response status: {response.status_code}. Error: \n{response.text}\n"
        )

    res = response.json()

    # # Here you would actually make an API call to login. For now, we'll simulate.
    save_state(
        {
            TOKEN_KEY: res[TOKEN_KEY],
            "url": f"{args.host}:{args.port}",
            "email": args.email,
        },
        args.dir,
        print_result,
    )

    if print_result:
        print("\nSignup & authentication successful. You may now use the Bystro API!\n")

    return res


def login(args: argparse.Namespace, print_result=True):
    if print_result:
        print(f"\nLogging into {args.host}:{args.port} with email: {args.email}.")

    url = f"{args.host}:{args.port}/api/user/auth/local"

    body = {"email": args.email, "password": args.password}

    response = requests.post(url, data=body)

    if response.status_code != 200:
        raise RuntimeError(
            f"Login failed with response status: {response.status_code}. Error: \n{response.text}\n"
        )

    res = response.json()

    save_state(
        {
            TOKEN_KEY: res[TOKEN_KEY],
            "url": args.host + ":" + str(args.port),
            "email": args.email,
        },
        args.dir,
        print_result,
    )

    if print_result:
        print("\nLogin successful. You may now use the Bystro API!\n")

    return res


def authenticate(args):
    state = load_state(args.dir)
    url = state.get("url")
    token = state.get(TOKEN_KEY)
    email = state.get("email")

    if not (url and token and email):
        raise ValueError("\n\nYou are not logged in. Please login first.\n")

    header = {"Authorization": f"Bearer {token}"}
    return url, header, email


def get_jobs(args: argparse.Namespace, print_result=True):
    url, auth_header, _ = authenticate(args)
    url = url + "/api/jobs"
    job_type = args.type
    job_id = args.id

    if not (job_id or job_type):
        raise ValueError("Please specify either a job id or a job type")

    if job_id and job_type:
        raise ValueError("Please specify either a job id or a job type, not both")

    if not job_id and job_type not in JOB_TYPE_ROUTE_MAP.keys():
        raise ValueError(
            f"Invalid job type: {job_type}. Valid types are: {','.join(JOB_TYPE_ROUTE_MAP.keys())}"
        )

    url = url + f"/{job_id}" if job_id else url + JOB_TYPE_ROUTE_MAP[job_type]

    if print_result:
        if job_id:
            print(f"\nFetching job with id:\t{job_id}")
        else:
            print(f"\nFetching jobs of type:\t{job_type}")

    response = requests.get(url, headers=auth_header)

    if response.status_code != 200:
        raise RuntimeError(
            f"Fetching jobs failed with response status: {response.status_code}. Error: {response.text}"
        )

    jobs = response.json()

    if job_id:
        # As mongo doesn't support keys with '.' in them, we need to store the config as a string
        jobs["config"] = json.loads(jobs["config"])

    if print_result:
        print("\nJob(s) fetched successfully: \n")
        print(json.dumps(jobs, indent=4))
        print("\n")

    return jobs


def create_job(args: argparse.Namespace, print_result=True):
    url, auth_header, _ = authenticate(args)
    url = url + "/api/jobs/upload/"

    payload = {
        "job": json.dumps(
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
                (os.path.basename(file), open(file, "rb"), "application/octet-stream"), # noqa: SIM115
            )
        )

    if print_result:
        print(f"\nCreating jobs for files: {','.join(map(lambda x: x[1][0], files))}\n")

    response = requests.post(url, headers=auth_header, data=payload, files=files)

    if response.status_code != 200:
        raise RuntimeError(
            f"Job creation failed with response status: {response.status_code}.\
                Error: \n{response.text}\n"
        )

    if print_result:
        print("\nJob creation successful:\n")
        print(json.dumps(response.json(), indent=4))
        print("\n")

    return response.json()


def get_user(args, print_result=True):
    if print_result:
        print("\n\nFetching user profile\n")

    res = authenticate(args)

    url, auth_header, email = res

    response = requests.request("GET", url + "/api/user/me", headers=auth_header)

    if response.status_code != 200:
        raise RuntimeError(
            f"Fetching profile failed with response status: {response.status_code}.\
                Error: \n{response.text}\n"
        )

    if print_result:
        print(f"\nFetched Profile for email {email}\n")
        print(json.dumps(response.json(), indent=4))
        print("\n")

    return response.json()


def main():
    parser = argparse.ArgumentParser(
        prog="bystro-api", description="Bystro CLI tool for making API calls."
    )
    subparsers = parser.add_subparsers(title="commands")

    # Adding the user sub-command
    login_parser = subparsers.add_parser("login", help="Login to the system")
    login_parser.add_argument("--host", required=True, help="Host of the system")
    login_parser.add_argument(
        "--port", type=int, default=443, help="Port of the system"
    )
    login_parser.add_argument("--email", required=True, help="Email to login with")
    login_parser.add_argument("--password", required=True, help="Password to login")
    login_parser.add_argument(
        "--dir", default="./", help="Where to save Bystro API login state"
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
    signup_parser.add_argument("--host", required=True, help="Host of the system")
    signup_parser.add_argument(
        "--port", type=int, default=443, help="Port of the system"
    )
    signup_parser.add_argument(
        "--dir", default="./", help="Where to save Bystro API login state"
    )
    signup_parser.set_defaults(func=signup)

    user_parser = subparsers.add_parser("get-user", help="Handle user operations")
    user_parser.add_argument("--profile", action="store_true", help="Get user profile")
    user_parser.add_argument(
        "--dir", default="./", help="Where Bystro API login state is saved"
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
        "--index",
        type=bool,
        default=True,
        help="Whether or not to index the annotation",
    )
    create_jobs_parser.add_argument(
        "--dir", default="./", help="Where Bystro API login state is saved"
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
        "--dir", default="./", help="Where Bystro API login state is saved"
    )
    jobs_parser.set_defaults(func=get_jobs)

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
