from bystro.cli.proteomics_cli import upload_proteomics_cli


def test_upload_proteomics_cli(mocker):
    mocker.patch(
        "bystro.cli.proteomics_cli.upload_proteomics_dataset", return_value="proteomics_dataset_result"
    )

    args = mocker.Mock(
        protein_abundance_file="./abundance_gene_MD.tsv",
        experiment_annotation_file=None,
        annotation_job_id="12345",
        proteomics_dataset_type="fragpipe-TMT",
    )

    result = upload_proteomics_cli(args)

    assert result == "proteomics_dataset_result"
