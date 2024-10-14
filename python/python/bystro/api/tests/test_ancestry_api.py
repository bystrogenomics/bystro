from pathlib import Path
import os

import pandas as pd

import pytest

from bystro.ancestry.ancestry_types import AncestryResults
from bystro.ancestry.inference import AncestryModels
from bystro.ancestry.tests.test_inference import (
    ANCESTRY_MODEL,
)

from bystro.api.ancestry import calculate_ancestry_scores, ancestry_json_to_format


pd.options.future.infer_string = True  # type: ignore


@pytest.mark.integration("Requires bystro-vcf to be installed as well as AWS credentials.")
def test_calculate_ancestry_scores_happy_path(mocker, tmpdir):
    mocker.patch(
        "bystro.ancestry.model.get_models",
        return_value=AncestryModels(ANCESTRY_MODEL, ANCESTRY_MODEL),
    )

    VCF_PATH = Path(__file__).parent / "trio.trim.vep.vcf.gz"
    ancestry_response = calculate_ancestry_scores(
        str(VCF_PATH), "hg19", dosage=False, out_dir=str(tmpdir)
    )

    assert isinstance(ancestry_response, AncestryResults)

    # Demonstrate that all expected sample_ids are accounted for
    samples_seen = set()
    expected_samples = set(["1805", "4805", "1847"])
    for result in ancestry_response.results:
        samples_seen.add(result.sample_id)

    assert samples_seen == expected_samples

def test_ancestry_tsv(tmp_path):
    pwd = os.path.dirname(os.path.abspath(__file__))
    ancestry_file_path = Path(pwd) / "ancestry_input.json"
    expected_results_path = Path(pwd) / "ancestry_expected_output.tsv"

    expected = pd.read_csv(expected_results_path, sep="\t")
    print("expected", expected)

    # create tmp file
    output_tsv_path = tmp_path / "output.tsv"
    output_csv_path = tmp_path / "output.csv"

    # Run the conversion for TSV
    ancestry_json_to_format(ancestry_file_path, output_tsv_path, "tsv")
    df1 = pd.read_csv(output_tsv_path, sep="\t")

    # Run the conversion for Excel
    ancestry_json_to_format(ancestry_file_path, output_csv_path, "csv")
    df2 = pd.read_csv(output_csv_path)

    assert expected.equals(df1), "TSV files do not match"
    assert expected.equals(df2), "CSV files do not match"
