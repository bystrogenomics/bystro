from argparse import Namespace
from pathlib import Path

from bystro.api.tests.test_cli import EXAMPLE_CACHED_AUTH
from bystro.proteomics.proteomics_cli import _configure_parser, upload_proteomics_dataset
from msgspec import json as mjson


def test_upload_proteomics_dataset(mocker):
    mocker.patch(
        "bystro.proteomics.proteomics_cli.authenticate",
        return_value=(
            EXAMPLE_CACHED_AUTH,
            "localhost:8080",
        ),
    )

    mock_response = '{"success": true}'
    mocker.patch(
        "requests.post",
        return_value=mocker.Mock(
            status_code=200, json=lambda: mjson.decode(mock_response), text=mock_response
        ),
    )

    protein_abundance_filename = str(Path(__file__).parent / "protein_abundance_file.tsv")
    experiment_annotation_filename = str(Path(__file__).parent / "experiment_annotation_file.tsv")

    args = Namespace(
        protein_abundance_file=protein_abundance_filename,
        experiment_annotation_file=experiment_annotation_filename,
        dir="./",
    )
    response = upload_proteomics_dataset(args)

    assert response == {"success": True}


def test__configure_parser():
    parser = _configure_parser()
    help_message = parser.format_help()
    expected_commands = "{login,upload-proteomics-dataset}"

    assert expected_commands in help_message
