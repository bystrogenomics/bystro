from unittest.mock import patch, MagicMock

import pandas as pd
import pyarrow as pa  # type: ignore
import pytest
from bystro.prs.preprocess_for_prs import (
    _load_association_scores,
    _load_genetic_maps_from_feather,
    _extract_nomiss_dosage_loci,
    _preprocess_scores,
    calculate_abs_effect_weights,
    compare_alleles,
    get_p_value_thresholded_indices,
    find_bin_for_row,
    generate_c_and_t_prs_scores,
    select_max_effect_per_bin,
)

AD_SCORE_FILEPATH = "fake_file.txt"


@pytest.fixture()
def mock_scores_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SNPID": ["1:566875:C:T", "1:728951:A:G"],
            "CHR": [1, 1],
            "POS": [566875, 728951],
            "OTHER_ALLELE": ["C", "G"],
            "EFFECT_ALLELE": ["T", "A"],
            "P": [0.699009, 0.030673],
            "BETA": [0.007630, -0.020671],
        }
    )


@pytest.fixture()
def mock_processed_scores_df() -> pd.DataFrame:
    mock_gwas_data = {
        "CHR": [1, 1],
        "POS": [566875, 728951],
        "OTHER_ALLELE": ["C", "G"],
        "EFFECT_ALLELE": ["T", "A"],
        "P": [0.699009, 0.030673],
        "SNPID": ["1:566875:C:T", "1:728951:A:G"],
        "BETA": [0.007630, -0.020671],
        "ID_effect_as_alt": ["chr1:566875:C:T", "chr1:728951:G:A"],
        "ID_effect_as_ref": ["chr1:566875:T:C", "chr1:728951:A:G"],
    }
    return pd.DataFrame(mock_gwas_data)


@pytest.fixture()
def mock_dosage_df():
    return pd.DataFrame(
        {
            "locus": ["chr1:566875:C:T", "chr1:728951:A:G", "chr1:917492:C:T", "chr2:917492:A:T"],
            "ID00096": [1, 1, 2, -1],
            "ID00097": [0, 1, 1, 0],
        }
    )


@pytest.fixture()
def mock_dosage_df_clean():
    return pd.DataFrame(
        {
            "locus": ["chr1:566875:C:T", "chr1:728951:A:G", "chr1:917492:C:T"],
            "ID00096": [1, 1, 2],
            "ID00097": [0, 1, 1],
        }
    )


@pytest.fixture()
def mock_genetic_maps():
    genetic_map_data = {
        "GeneticMap1": pd.DataFrame({"upper_bound": [1000, 2000, 3000]}),
    }
    return genetic_map_data


@pytest.fixture()
def mock_bin_mappings():
    return {
        1: [1000, 2000, 3000],
    }


@pytest.fixture()
def mock_final_dosage_scores():
    genos_transpose_mock = pd.DataFrame(
        {
            "chr10:10082621:G:T": [0, 2],
            "chr10:10528970:A:G": [1, 0],
        },
        index=["Sample1", "Sample2"],
    )

    scores_overlap_adjusted_mock = pd.DataFrame(
        {
            "SNPID": ["chr10:10082621:G:T", "chr10:10528970:A:G"],
            "BETA": [0.1, -0.2],
        }
    ).set_index("SNPID")

    expected_results = pd.Series(
        {
            "Sample1": (0 * 0.1) + (1 * -0.2),
            "Sample2": (2 * 0.1) + (0 * -0.2),
        }
    )
    return genos_transpose_mock, scores_overlap_adjusted_mock, expected_results


def test_extract_nomiss_dosage_loci(mock_dosage_df):
    filtered_df = mock_dosage_df[mock_dosage_df["ID00096"] != -1]
    mock_table = pa.Table.from_pandas(filtered_df)
    with patch("pyarrow.dataset.dataset") as mock_dataset:
        mock_ds = MagicMock()
        mock_ds.schema.names = mock_dosage_df.columns.tolist()
        mock_ds.filter.return_value = mock_ds
        mock_ds.to_table.return_value = mock_table
        mock_dataset.return_value = mock_ds
        result = _extract_nomiss_dosage_loci("mock/path/to/dosage.feather")

        assert not result.empty, "The DataFrame should not be empty."
        assert "locus" in result.columns, "'locus' column should be present in the DataFrame."
        assert (
            len(result) == 3
        ), "The number of rows in the DataFrame should be 3."
        mock_dataset.assert_called_once_with("mock/path/to/dosage.feather", format="feather")


def test_load_genetic_maps_from_feather(tmp_path):
    test_dir = tmp_path / "ProcessedGeneticMaps"
    test_dir.mkdir()
    test_file = test_dir / "chromosome_1_genetic_map.feather"
    mock_map = pd.DataFrame({"upper_bound": [1000, 2000, 3000], "chromosome_num": [1, 1, 1]})
    mock_map.to_feather(test_file)
    genetic_maps = _load_genetic_maps_from_feather(str(test_dir))
    assert isinstance(genetic_maps, dict), "The function should return a dictionary."
    assert "GeneticMap1" in genetic_maps, "The dictionary should contain keys in the expected format."
    assert not genetic_maps["GeneticMap1"].empty, "The DataFrame for chr 1 should not be empty."
    assert len(genetic_maps) == 1, "There should be exactly one genetic map loaded."
    assert all(
        column in genetic_maps["GeneticMap1"].columns for column in mock_map.columns
    ), "The DataFrame should contain the expected columns."


@patch("pyarrow.feather.read_feather")
def test_load_scores(mock_read_feather, mock_scores_df: pd.DataFrame):
    # Reset index if your actual function expects 'SNPID' as a column, not an index
    mock_read_feather.return_value = mock_scores_df.reset_index()
    result_df = _load_association_scores("fake_file_path.feather")
    expected_columns = ["CHR", "POS", "OTHER_ALLELE", "EFFECT_ALLELE", "P", "SNPID", "BETA"]
    assert not result_df.empty, "The DataFrame should not be empty."
    assert (
        list(result_df.columns) == expected_columns
    ), "DataFrame columns do not match expected columns."
    assert len(result_df) == len(mock_scores_df), "DataFrame length does not match expected length."


@patch("bystro.prs.preprocess_for_prs._load_association_scores")
def test_preprocess_scores(mock_load_association_scores, mock_scores_df: pd.DataFrame):
    mock_load_association_scores.return_value = mock_scores_df
    processed_scores = _preprocess_scores(AD_SCORE_FILEPATH)

    assert "SNPID" in processed_scores.columns
    assert "ID_effect_as_alt" in processed_scores.columns
    assert "ID_effect_as_ref" in processed_scores.columns


def test_get_p_value_thresholded_indices(mock_processed_scores_df: pd.DataFrame):
    p_value_threshold = 0.05
    thresholded_indices = get_p_value_thresholded_indices(mock_processed_scores_df, p_value_threshold)
    filtered_scores = mock_processed_scores_df.loc[thresholded_indices]
    assert (
        len(filtered_scores) == 1
    ), "Filtered scores should only contain 1 row for p_value_threshold=0.05."
    assert all(
        filtered_scores["P"] < p_value_threshold
    ), "All rows should have P-values less than the threshold."
    expected_snps = {"1:728951:A:G"}
    assert (
        set(filtered_scores["SNPID"]) == expected_snps
    ), "Filtered scores should contain expected SNP(s)."


def test_compare_alleles():
    data = {
        "pair1": ["chr1:100:A:T", "chr1:200:G:A", "chr1:400:G:T"],
        "pair2": [
            "chr1:100:A:T",  # direct match
            "chr1:200:A:G",  # reversed alt/ref match
            "chr1:400:C:G",  # no match
        ],
    }
    test_alleles = pd.DataFrame(data)
    assert (
        compare_alleles(test_alleles.iloc[0], "pair1", "pair2") == "Direct Match"
    ), "Direct match test failed"
    assert (
        compare_alleles(test_alleles.iloc[1], "pair1", "pair2") == "Effect Allele Is Ref"
    ), "Reversed match test failed"
    assert (
        compare_alleles(test_alleles.iloc[2], "pair1", "pair2") == "Alleles Do Not Match"
    ), "No match test failed"


def test_find_bin_for_row(mock_bin_mappings: dict[int, list[int]]):
    row1 = pd.Series({"CHR": "1", "POS": 1500})
    expected1 = 1
    result1 = find_bin_for_row(row1, mock_bin_mappings)
    assert result1 == expected1, "Failed test: Valid chromosome and position"

    row2 = pd.Series({"CHR": "3", "POS": 1000})
    expected2 = None
    result2 = find_bin_for_row(row2, mock_bin_mappings)
    assert result2 == expected2, "Failed test: Chromosome not in bin_mappings"

    row3 = pd.Series({"CHR": "invalid", "POS": 1000})
    expected3 = None
    result3 = find_bin_for_row(row3, mock_bin_mappings)
    assert result3 == expected3, "Failed test: Invalid chromosome value"

    row4 = pd.Series({"CHR": "1", "POS": 0})
    expected4 = 0
    result4 = find_bin_for_row(row4, mock_bin_mappings)
    assert result4 == expected4, "Failed test: First bin"


def test_calculate_abs_effect_weights(mock_processed_scores_df):
    result_df = calculate_abs_effect_weights(mock_processed_scores_df)
    expected_abs_values = [0.007630, 0.020671]  # Expected absolute values
    calculated_abs_values = result_df["abs_effect_weight"].tolist()
    assert (
        calculated_abs_values == expected_abs_values
    ), "The calculated and expected values do not match."


def test_select_max_effect_per_bin():
    mock_data = {
        "CHR": [1, 1, 2, 2, 2],
        "bin": [1, 1, 2, 2, 2],
        "abs_effect_weight": [0.2, 0.5, 0.3, 0.4, 0.1],
    }
    mock_scores_genos_df = pd.DataFrame(mock_data)
    result_df = select_max_effect_per_bin(mock_scores_genos_df)
    expected_results = {
        (1, 1): 0.5,
        (2, 2): 0.4,
    }
    for chr_bin, expected_max in expected_results.items():
        chrom, bin_id = chr_bin
        assert (
            result_df[(result_df["CHR"] == chrom) & (result_df["bin"] == bin_id)][
                "abs_effect_weight"
            ].iloc[0]
            == expected_max
        )
    assert len(result_df) == len(
        expected_results
    ), "The result DataFrame should have one row per (CHR, bin) group."


@patch("bystro.prs.preprocess_for_prs.finalize_dosage_scores_after_c_t")
def test_generate_c_and_t_prs_scores(mock_finalize_dosage_scores_after_c_t, mock_final_dosage_scores):
    genos_transpose_mock, scores_overlap_adjusted_mock, expected_results = mock_final_dosage_scores

    def mock_finalize_func(*_, **__):
        return genos_transpose_mock, scores_overlap_adjusted_mock

    mock_finalize_dosage_scores_after_c_t.side_effect = mock_finalize_func
    prs_results = generate_c_and_t_prs_scores("dummy_gwas_path", "dummy_dosage_path", "dummy_map_path")
    prs_series = pd.Series(prs_results, index=genos_transpose_mock.index)
    pd.testing.assert_series_equal(prs_series, expected_results, check_dtype=False, check_names=False)
