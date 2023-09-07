from typing import Tuple, BinaryIO
import argparse
import os
import requests

from msgspec import json as mjson

from bystro.api.cli import DEFAULT_DIR, authenticate, login

UPLOAD_PROTEIN_ENDPOINT = "/api/jobs/upload_protein/"


def _package_filename(filename: str) -> Tuple[str, Tuple[str, BinaryIO, str]]:
    """Wrap filename in a container suitable for upload through the requests library."""
    return (
        "file",
        (
            os.path.basename(filename),
            open(filename, "rb"),  # noqa: SIM115
            "application/octet-stream",
        ),
    )


def upload_proteomics_dataset(args: argparse.Namespace, print_result=True) -> dict:
    """Upload a proteomics dataset (consisting of a protein abundance
    file and an experiment annnotation file) through the /api/jobs/upload_protein endpoint.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command.
    print_result : bool, optional
        Whether to print the result of the upload operation, by default True.

    Returns
    -------
    dict
        A json response from the endpoint.

    """
    state, auth_header = authenticate(args)
    url = os.path.join(state.url, UPLOAD_PROTEIN_ENDPOINT)

    payload = {
        "job": mjson.encode(
            {
                "protein_abundance_file": os.path.basename(args.protein_abundance_file),
                "experiment_annotation_file": os.path.basename(args.experiment_annotation_file),
                "proteomics_dataset_type": "fragpipe-TMT",  # we currently only support this format
            }
        )
    }

    files_to_upload = [args.protein_abundance_file, args.experiment_annotation_file]
    files = [_package_filename(filename) for filename in files_to_upload]

    if print_result:
        print(f"\nCreating jobs for files: {', '.join(files_to_upload)}\n")

    response = requests.post(url, headers=auth_header, data=payload, files=files, timeout=30)

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


def _add_login_subparser(subparsers) -> None:
    """Add subparser for login command."""
    login_parser = subparsers.add_parser("login", help="Authenticate with the Bystro API")
    login_parser.add_argument(
        "--host",
        required=True,
        help="Host of the Bystro API server, e.g. https://bystro-dev.emory.edu",
    )
    login_parser.add_argument(
        "--port", type=int, default=443, help="Port of the Bystro API server, e.g. 443"
    )
    login_parser.add_argument("--email", required=True, help="Email to login with")
    login_parser.add_argument("--password", required=True, help="Password to login with")
    login_parser.add_argument("--dir", default=DEFAULT_DIR, help="Where to save Bystro API login state")
    login_parser.set_defaults(func=login)


def _add_upload_proteomics_dataset_subparser(subparsers) -> None:
    """Add subparser for upload_proteomics_dataset command."""
    upload_proteomics_dataset_parser = subparsers.add_parser(
        "upload-proteomics-dataset", help="Upload a Fragpipe TMT proteomics dataset"
    )
    upload_proteomics_dataset_parser.add_argument(
        "--protein-abundance-file",
        required=True,
        type=str,
        help="Protein abundance file (currently only Fragpipe TMT .tsv's accepted.)",
    )
    upload_proteomics_dataset_parser.add_argument(
        "--experiment-annotation-file",
        required=True,
        type=str,
        help="Experiment annotation file (currently only Fragpipe format accepted.)",
    )

    upload_proteomics_dataset_parser.add_argument(
        "--dir", default=DEFAULT_DIR, help="Where Bystro API login state is saved"
    )
    upload_proteomics_dataset_parser.set_defaults(func=upload_proteomics_dataset)


def _configure_parser() -> argparse.ArgumentParser:
    """Configure parser for command line arguments."""
    parser = argparse.ArgumentParser(
        prog="bystro-proteomics-api",
        description="Bystro CLI tool for interacting with proteomics service.",
    )
    subparsers = parser.add_subparsers(title="commands")
    _add_login_subparser(subparsers)
    _add_upload_proteomics_dataset_subparser(subparsers)

    return parser


def main():
    parser = _configure_parser()
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
