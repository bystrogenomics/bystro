"""Provide a CLI for proteomics analysis."""

import json
from pathlib import Path
from typing import Any, BinaryIO
import uuid

import requests

from bystro.api.auth import authenticate
from bystro.proteomics.somascan import SomascanDataset

# ruff: noqa: T201

UPLOAD_PROTEIN_ENDPOINT = "/api/jobs/proteomics/"
GET_ANNOTATION = "/api/jobs/:id"
EXPERIMENT_ENDPOINT = "/api/jobs/experiment"

HTTP_STATUS_OK = 200
ONE_HOUR_IN_SECONDS = 60 * 60
FRAGPIPE_GENE_ABUNDANCE_HEADERS = ["Index", "NumberPSM", "ProteinID", "MaxPepProb", "ReferenceIntensity"]
EXPERIMENT_ANNOTATION_HEADERS = ["Experiment Name", "Sample ID", "Subject ID"]
SOMASCAN_EXTENSION = ".adat"


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


def upload_proteomics_dataset(
    protein_abundance_file: str,
    experiment_annotation_file: str | None = None,
    annotation_job_id: str | None = None,
    experiment_name: str | None = None,
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
    annotation_job_id : str | None
        annotationID of the job associated with the annotation dataset.
    experiment_name : str | None
        Name of the experiment, required if the experiment annotation file contains multiple experiments
    print_result : bool
        Whether to print the result of the upload operation, by default True.

    Returns
    -------
    dict
        A json response with annotationID and proteomicsID.
    """

    if annotation_job_id is not None and not isinstance(annotation_job_id, str):
        raise ValueError("annotation job id must be a string.")

    if not protein_abundance_file.endswith(SOMASCAN_EXTENSION):
        with open(protein_abundance_file, "r", encoding="utf8") as file:
            first_line = file.readline().strip().lower()
            if not all(header.lower() in first_line for header in FRAGPIPE_GENE_ABUNDANCE_HEADERS):
                raise ValueError(
                    "The protein abundance file does not contain the expected headers: %s"
                    % FRAGPIPE_GENE_ABUNDANCE_HEADERS
                )
    else:
        try:
            SomascanDataset.from_paths(protein_abundance_file)
        except ValueError as e:
            raise ValueError("Failed to read the somascan dataset") from e

    if experiment_annotation_file:
        with open(experiment_annotation_file, "r", encoding="utf8") as file:
            first_line = file.readline().strip().lower()
            if not all(header.lower() in first_line for header in EXPERIMENT_ANNOTATION_HEADERS):
                raise ValueError(
                    "The experiment annotation file does not contain the expected headers: %s"
                    % EXPERIMENT_ANNOTATION_HEADERS
                )

            experiment_name_index = EXPERIMENT_ANNOTATION_HEADERS.index("Experiment Name")
            experiment_names = set()
            for line in file:
                row = line.strip().split("\t")
                experiment_names.add(row[experiment_name_index])

        if len(experiment_names) == 1:
            experiment_name = experiment_name or experiment_names.pop()
        elif not experiment_name:
            raise ValueError(
                "Either no experiment name or multiple experiment names "
                "were found in the experiment annotation file. "
                "Please provide an experiment name."
            )

        state, auth_header = authenticate()
        experiment_file = _package_filename(experiment_annotation_file)
        experiment_response = requests.post(
            state.url + EXPERIMENT_ENDPOINT,
            headers=auth_header,
            files={"file": experiment_file[1]},
            data={"experimentName": experiment_name},
            timeout=ONE_HOUR_IN_SECONDS,
        )

        if experiment_response.status_code != HTTP_STATUS_OK:
            raise RuntimeError(
                "Experiment annotation upload failed with status "
                f"code {experiment_response.status_code}: {experiment_response.text}"
            )

        if print_result:
            print(
                "\nExperiment Annotation Upload Response:",
                json.dumps(experiment_response.json(), indent=4),
            )

    state, auth_header = authenticate()
    url = state.url + UPLOAD_PROTEIN_ENDPOINT
    proteomics_uuid = str(uuid.uuid4())

    if annotation_job_id:
        annotation_url = state.url + GET_ANNOTATION.replace(":id", str(annotation_job_id))
        annotation_response = requests.get(annotation_url, headers=auth_header, timeout=30)

        if annotation_response.status_code != HTTP_STATUS_OK:
            raise RuntimeError(
                f"The annotation with ID {annotation_job_id} "
                "does not exist or you do not have permissions to access this annotation."
            )

        annotation_uuid = str(uuid.uuid4())

        job_payload = {
            "assembly": "NA",
            "annotationID": annotation_uuid,
            "proteomicsID": proteomics_uuid,
            "experimentName": experiment_name,
        }
    else:
        job_payload = {
            "assembly": "NA",
            "experimentName": experiment_name,
        }
    protein_files = [_package_filename(protein_abundance_file)]

    response = requests.post(
        url,
        headers=auth_header,
        files=protein_files,
        data={"job": json.dumps(job_payload)},
        timeout=ONE_HOUR_IN_SECONDS,
    )

    if response.status_code != HTTP_STATUS_OK:
        raise RuntimeError(f"Upload failed with status code {response.status_code}: {response.text}")

    proteomics_response_data = response.json()
    if print_result:
        print("\nProteomics Upload Response:", json.dumps(proteomics_response_data, indent=4))

    if annotation_job_id:
        update_annotation_payload = {"proteomicsID": proteomics_uuid, "annotationID": annotation_uuid}
        update_annotation_response = requests.post(
            annotation_url, headers=auth_header, json=update_annotation_payload, timeout=30
        )

        if update_annotation_response.status_code == HTTP_STATUS_OK:
            if print_result:
                annotation_name = annotation_response.json().get("name", "Unknown Annotation")
                print(
                    f'\nAnnotation "{annotation_name}" (annotationID: {annotation_uuid}) '
                    f'was linked to this proteomics submission (proteomicsID: "{proteomics_uuid}").\n'
                )
        else:
            raise RuntimeError(
                f"Annotation linkage failed with status code "
                f"{update_annotation_response.status_code}: {update_annotation_response.text}"
            )
    elif print_result:
        print("\nProteomics submission completed successfully without annotation linkage.\n")

    return proteomics_response_data