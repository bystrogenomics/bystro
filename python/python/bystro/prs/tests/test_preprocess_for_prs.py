from unittest.mock import patch

import pandas as pd
import pytest
from bystro.prs.preprocess_for_prs import (
    _load_association_scores,
    _load_genetic_maps_from_feather,
    _load_preprocessed_dosage_matrix,
    _preprocess_scores,
)

AD_SCORE_FILEPATH = 'fake_file.txt'

@pytest.fixture()
def mock_scores_df() -> pd.DataFrame:
    mock_gwas_data = {
        "CHR": [1, 1],
        "POS": [566875, 728951],
        "OTHER_ALLELE": ["C", "C"],
        "EFFECT_ALLELE": ["T", "T"],
        "P": [0.699009, 0.030673],
        "SNPID": ["1:566875:C:T", "1:728951:A:G"],
        "BETA": [0.007630, -0.020671],
    }
    return pd.DataFrame(mock_gwas_data)

@pytest.fixture()
def mock_dosage_df():
    return pd.DataFrame(
        {
            "locus": ["chr1:566875:C:T", "chr1:728951:A:G", "chr1:917492:C:T"],
            "ID00096": [1, 1, 2],
            "ID00097": [0, 1, 1],
        }
    )


def test_load_preprocessed_dosage_matrix(mock_dosage_df):
    with patch("pyarrow.feather.read_feather", return_value=mock_dosage_df) as mock_read:
        result = _load_preprocessed_dosage_matrix("mock/path/to/dosage.feather")
        assert not result.empty, "The DataFrame should not be empty."
        assert "locus" in result.columns, "'locus' column should be present in the DataFrame."
        assert len(result) == len(
            mock_dosage_df
        ), "The number of rows in the DataFrame should match the mock."
        mock_read.assert_called_once_with("mock/path/to/dosage.feather")


def test_load_genetic_maps_from_feather(tmp_path):
    test_dir = tmp_path / "ProcessedGeneticMaps"
    test_dir.mkdir()
    test_file = test_dir / "chromosome_1_genetic_map.feather"
    mock_map = pd.DataFrame({"upper_bound": [1000, 2000, 3000], "chromosome_num": [1, 1, 1]})
    mock_map.to_feather(test_file)
    genetic_maps = _load_genetic_maps_from_feather(str(test_dir))
    assert isinstance(genetic_maps, dict), "The function should return a dictionary."
    assert "GeneticMap1" in genetic_maps, "The dictionary should contain keys in the expected format."
    assert not genetic_maps["GeneticMap1"].empty, "The DataFrame for chromosome 1 should not be empty."
    assert len(genetic_maps) == 1, "There should be exactly one genetic map loaded."
    assert all(
        column in genetic_maps["GeneticMap1"].columns for column in mock_map.columns
    ), "The DataFrame should contain the expected columns."


@patch("pyarrow.feather.read_feather")
def test_load_scores(mock_read_feather, mock_scores_df: pd.DataFrame):
    mock_read_feather.return_value = mock_scores_df
    result_df = _load_association_scores("fake_file_path.feather")
    assert not result_df.empty, "The DataFrame should not be empty."
    assert list(result_df.columns) == list(
        mock_scores_df.columns
    ), "DataFrame columns do not match expected columns."
    assert len(result_df) == len(
        mock_scores_df["SNPID"]
    ), "DataFrame length does not match expected length."


@patch("bystro.prs.preprocess_for_prs._load_association_scores")
def test_preprocess_scores(mock_load_association_scores, mock_scores_df: pd.DataFrame):
    mock_load_association_scores.return_value = mock_scores_df
    processed_scores = _preprocess_scores(AD_SCORE_FILEPATH)
    
    assert "SNPID" in processed_scores.index.name
    assert "ID_effect_as_alt" in processed_scores.columns
    assert "ID_effect_as_ref" in processed_scores.columns
