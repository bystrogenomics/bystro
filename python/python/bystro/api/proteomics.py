"""Provide a CLI for proteomics analysis."""

from pathlib import Path
from typing import Any, BinaryIO

import requests
from bystro.api.auth import authenticate
from enum import Enum
import json

# ruff: noqa: T201

UPLOAD_PROTEIN_ENDPOINT = "/api/jobs/proteomics/"
GET_ANNOTATION = "/api/jobs/:id"
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


class DatasetTypes(Enum):
    fragpipe_TMT = "fragpipe-TMT"
    somascan = "somascan"


def upload_proteomics_dataset(
    protein_abundance_file: str,
    experiment_annotation_file: str | None = None,
    annotation_job_id: str | None = None,
    proteomics_dataset_type: DatasetTypes = DatasetTypes.fragpipe_TMT,
    print_result: bool = True,
) -> dict[str, Any]:
    """
    Upload a fragpipe-TMT dataset through the /api/jobs/proteomics/ endpoint and
    update the annotation job.

    Parameters
    ----------
    protein_abundance_file : str
        Path to the protein abundance file.
    experiment_annotation_file : str | None
        Path to the experiment annotation file.
    experiment_annotation_file : str | None
        ID of the job associated with the annotation dataset.
    proteomics_dataset_type : DatasetTypes
        Type of the proteomics dataset (we only support fragpipe-TMT currently).
    print_result : bool
        Whether to print the result of the upload operation, by default True.

    Returns
    -------
    dict
        A json response with annotationID and proteomicsID.
    """

    if annotation_job_id is not None and not isinstance(annotation_job_id, str):
        raise ValueError("annotation job id must be a string.")

    abundance_required_headers = ["Index", "NumberPSM", "ProteinID", "MaxPepProb", "ReferenceIntensity"]
    with open(protein_abundance_file, "r") as file:
        first_line = file.readline().strip().lower()
        if not all(header.lower() in first_line for header in abundance_required_headers):
            if print_result:
                print("Error: The protein abundance file does not contain the required headers.")
            return {}

    state, auth_header = authenticate()
    url = state.url + UPLOAD_PROTEIN_ENDPOINT

    if annotation_job_id:
        annotation_url = state.url + GET_ANNOTATION.replace(":id", str(annotation_job_id))
        annotation_response = requests.get(annotation_url, headers=auth_header)

        if annotation_response.status_code != HTTP_STATUS_OK:
            print(
                f"The annotation with ID {annotation_job_id} "
                "does not exist or you do not have permissions to access this annotation."
            )
            return {}

    job_payload = {
        "protein_abundance_file": Path(protein_abundance_file).name,
        "proteomics_dataset_type": proteomics_dataset_type.value,
        "assembly": "N/A",
    }

    files = [_package_filename(protein_abundance_file)]
    if experiment_annotation_file:
        files.append(_package_filename(experiment_annotation_file))

    response = requests.post(
        url, headers=auth_header, files=files, data={"job": json.dumps(job_payload)}
    )

    if response.status_code != HTTP_STATUS_OK:
        raise Exception(f"Upload failed with status code {response.status_code}: {response.text}")

    proteomics_response_data = response.json()
    if print_result:
        print("\nProteomics Upload response:", json.dumps(proteomics_response_data, indent=4))

    proteomics_job_id = proteomics_response_data.get("_id")
    final_response = {"proteomicsID": proteomics_job_id}

    if annotation_job_id:
        update_annotation_payload = {"proteomicsID": proteomics_job_id}
        update_annotation_response = requests.patch(
            annotation_url, headers=auth_header, json=update_annotation_payload
        )

        if update_annotation_response.status_code == HTTP_STATUS_OK:
            final_response["annotationID"] = annotation_job_id
            if print_result:
                annotation_name = annotation_response.json().get("inputFileName", "Unknown Annotation")
                print(
                    f'\nAnnotation "{annotation_name}" (id: {annotation_job_id}) '
                    f'was linked to this proteomics submission (id: "{proteomics_job_id}").\n'
                )
    elif print_result:
        print("\nProteomics submission completed successfully without annotation linkage.\n")

    if print_result:
        print(json.dumps(final_response, indent=4))

    return final_response
