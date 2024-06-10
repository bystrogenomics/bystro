from pathlib import Path

from bystro.cli.proteomics_cli import upload_proteomics_cli

ABUNDANCE_FILE_PATH = str(
    Path(__file__).parent.parent.parent / "proteomics" / "tests" / "example_abundance_gene_MD.tsv"
)


def test_upload_proteomics_cli(mocker):
    mocker.patch(
        "bystro.cli.proteomics_cli.upload_proteomics_dataset", return_value="proteomics_dataset_result"
    )

    args = mocker.Mock(
        protein_abundance_file=ABUNDANCE_FILE_PATH,
        experiment_annotation_file=None,
        annotation_job_id="12345",
    )

    result = upload_proteomics_cli(args)

    assert result == "proteomics_dataset_result"
