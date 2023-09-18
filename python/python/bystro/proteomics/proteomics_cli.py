"""Provide a CLI for proteomics analysis."""
import argparse
from pathlib import Path
from typing import Any, BinaryIO

import requests
from bystro.api.cli import DEFAULT_DIR, authenticate, login
from msgspec import json as mjson

# ruff: noqa: T201

UPLOAD_PROTEIN_ENDPOINT = "/api/jobs/upload_protein/"
HTTP_STATUS_OK = 200
ONE_HOUR_IN_SECONDS = 60 * 60


def _package_filename(filename: str) -> tuple[str, tuple[str, BinaryIO, str]]:
    """Wrap filename in a container suitable for upload through the requests library."""
    filepath = Path(filename)
    return (
        "file",
        (
            filepath.name,
            filepath.open("rb"),
            "application/octet-stream",
        ),
    )


def upload_proteomics_dataset(args: argparse.Namespace, *, print_result: bool = True) -> dict[str, Any]:
    """Upload a fragpipe-TMT dataset through the /api/jobs/upload_protein endpoint.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed to the command
    print_result : bool, optional
        Whether to print the result of the upload operation, by default True.

    Returns
    -------
    dict
        A json response from the endpoint.

    """
    state, auth_header = authenticate(args)
    url = str(state.url / Path(UPLOAD_PROTEIN_ENDPOINT))

    payload = {
        "job": mjson.encode(
            {
                "protein_abundance_file": Path(args.protein_abundance_file).name,
                "experiment_annotation_file": Path(args.experiment_annotation_file).name,
                "proteomics_dataset_type": "fragpipe-TMT",  # we currently only support this format
            }
        )
    }

    files_to_upload = [args.protein_abundance_file, args.experiment_annotation_file]
    files = [_package_filename(filename) for filename in files_to_upload]

    if print_result:
        print(f"\nCreating jobs for files: {', '.join(files_to_upload)}\n")

    response = requests.post(
        url, headers=auth_header, data=payload, files=files, timeout=ONE_HOUR_IN_SECONDS
    )

    if response.status_code != HTTP_STATUS_OK:
        msg = (
            f"Job creation failed with response status: {response.status_code}.  "
            f"Error: \n{response.text}\n"
        )
        raise RuntimeError(msg)

    if print_result:
        print("\nJob creation successful:\n")
        print(mjson.format(response.text, indent=4))
        print("\n")

    response_json: dict[str, Any] = response.json()
    return response_json


#  subparsers is a public class of argparse but is named with a
#  leading underscore, which is a design flaw.  The noqas on the helper methods below
#  below suppress the warnings about this.


def _add_login_subparser(subparsers: argparse._SubParsersAction) -> None:  # noqa: SLF001
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


def _add_upload_proteomics_dataset_subparser(
    subparsers: argparse._SubParsersAction,  # noqa: SLF001
) -> None:
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


def main() -> None:
    """Run the proteomics CLI."""
    parser = _configure_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        try:
            args.func(args)
        except Exception as e:  # noqa: BLE001
            print(f"\nSomething went wrong:\t{e}\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
