from unittest.mock import patch
import json

from msgspec import json as mjson
import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore
import pytest
from bystro.ancestry.ancestry_types import PopulationVector, AncestryResults
from bystro.prs.preprocess_for_prs import (
    _load_association_scores,
    _load_genetic_maps_from_feather,
    _preprocess_scores,
    calculate_abs_effect_weights,
    compare_alleles,
    find_bin_for_row,
    generate_c_and_t_prs_scores,
    select_max_effect_per_bin,
)

pd.options.future.infer_string = True  # type: ignore

AD_SCORE_FILEPATH = "fake_file.txt"


@pytest.fixture()
def mock_population_allele_frequencies():
    return pd.DataFrame(
        {
            "locus": ["chr8:132782505:T:C", "chr3:183978846:A:G"],
            "AFR": [0.0460780002176762, 0.158682003617287],
            "AMR": [0.0814258009195328, 0.273122996091843],
            "EAS": [0.0251158997416496, 0.713967978954315],
            "EUR": [0.156823992729187, 0.303905993700027],
            "SAS": [0.108804002404213, 0.559723019599915],
        }
    ).set_index("locus")


@pytest.fixture()
def mock_scores_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CHR": [1, 1],
            "POS": [566875, 728951],
            "OTHER_ALLELE": ["C", "G"],
            "EFFECT_ALLELE": ["T", "A"],
            "P": [0.699009, 0.0030673],
            "BETA": [0.007630, -0.020671],
            "EFFECT_ALLELE_FREQUENCY": [0.1, 0.2],
            "SNPID": ["chr8:132782505:T:C", "chr3:183978846:A:G"]
        },
        index=["chr8:132782505:T:C", "chr3:183978846:A:G"],
    )


@pytest.fixture()
def mock_processed_scores_df() -> pd.DataFrame:
    mock_gwas_data = {
        "CHR": [1, 1],
        "POS": [566875, 728951],
        "OTHER_ALLELE": ["C", "G"],
        "EFFECT_ALLELE": ["T", "A"],
        "P": [0.699009, 0.0030673],
        "BETA": [0.007630, -0.020671],
        "EFFECT_ALLELE_FREQUENCY": [0.1, 0.2],
        "SNPID": ["chr8:132782505:T:C", "chr3:183978846:A:G"],
        "ID_effect_as_alt": ["chr8:132782505:T:C", "chr3:183978846:A:G"],
        "ID_effect_as_ref": ["chr8:132782505:C:T", "chr3:183978846:G:A"],
    }
    return pd.DataFrame(mock_gwas_data, index=["chr8:132782505:T:C", "chr3:183978846:A:G"])


@pytest.fixture()
def mock_scores_loci(mock_processed_scores_df: pd.DataFrame) -> set:
    return set(mock_processed_scores_df["SNPID"].tolist())


@pytest.fixture()
def mock_dosage_df():
    return pd.DataFrame(
        {
            "locus": [
                "chr8:132782505:C:T",
                "chr3:183978846:A:G",
                "chr2:4000400:C:T",
                "chr21:24791946:C:T",
            ],
            "ID00096": [1, 1, 2, -1],
            "ID00097": [0, 1, 1, 0],
        }
    )


@pytest.fixture()
def mock_dosage_df_clean():
    return pd.DataFrame(
        {
            "locus": ["chr8:132782505:C:T", "chr3:183978846:A:G", "chr2:4000400:C:T"],
            "ID00096": [1, 1, 2],
            "ID00097": [0, 1, 1],
        }
    )


@pytest.fixture()
def mock_genetic_maps():
    genetic_map_data = {
        "GeneticMap1": pd.DataFrame({"upper_bound": [1000, 2000, 3000], "chromosome_num": [1, 1, 1]}),
        "GeneticMap2": pd.DataFrame({"upper_bound": [1000, 2000, 3000], "chromosome_num": [2, 2, 2]}),
    }
    return genetic_map_data


@pytest.fixture()
def mock_bin_mappings():
    return {
        1: [1000, 2000, 3000],
        2: [1000, 2000, 3000],
    }


@pytest.fixture()
def mock_finalize_dosage_after_c_t():
    def _mock_finalize_dosage_after_c_t(chunk, loci_and_allele_comparison):  # noqa: ARG001
        return pd.DataFrame(
            {"ID00096": [1, 1], "ID00097": [0, 1]}, index=["chr8:132782505:C:T", "chr3:183978846:G:A"]
        ).transpose()

    return _mock_finalize_dosage_after_c_t


def test_load_genetic_maps_from_feather(tmp_path, mock_genetic_maps):
    test_file = tmp_path / "combined_genetic_map.feather"

    combined_mock_map = pd.concat(mock_genetic_maps.values(), ignore_index=True)
    combined_mock_map.to_feather(test_file)
    genetic_maps = _load_genetic_maps_from_feather(str(test_file))

    assert isinstance(genetic_maps, dict), "The function should return a dictionary."
    assert "GeneticMap1" in genetic_maps, "The dictionary should contain keys in the expected format."
    assert "GeneticMap2" in genetic_maps, "The dictionary should contain keys in the expected format."
    assert not genetic_maps["GeneticMap1"].empty, "The DataFrame for chr 1 should not be empty."
    assert not genetic_maps["GeneticMap2"].empty, "The DataFrame for chr 2 should not be empty."
    assert len(genetic_maps) == 2, "There should be exactly two genetic maps loaded."
    assert all(
        column in genetic_maps["GeneticMap1"].columns for column in combined_mock_map.columns
    ), "The DataFrame should contain the expected columns."
    assert all(
        column in genetic_maps["GeneticMap2"].columns for column in combined_mock_map.columns
    ), "The DataFrame should contain the expected columns."


@patch("pyarrow.feather.read_feather")
def test_load_scores(mock_read_feather, mock_scores_df: pd.DataFrame):
    mock_read_feather.return_value = mock_scores_df.reset_index()
    result_df = _load_association_scores("fake_file_path.feather")
    print("result_df.columns", result_df.columns)
    expected_columns = [
        "CHR",
        "POS",
        "OTHER_ALLELE",
        "EFFECT_ALLELE",
        "P",
        "BETA",
        "EFFECT_ALLELE_FREQUENCY",
        "SNPID"
    ]

    assert not result_df.empty, "The DataFrame should not be empty."
    assert (
        list(result_df.columns) == expected_columns
    ), "DataFrame columns do not match expected columns."
    assert len(result_df) == len(mock_scores_df), "DataFrame length does not match expected length."


@patch("bystro.prs.preprocess_for_prs._load_association_scores")
def test_preprocess_scores(mock_load_association_scores, mock_scores_df: pd.DataFrame):
    mock_load_association_scores.return_value = mock_scores_df
    processed_scores = _preprocess_scores(mock_scores_df)

    assert processed_scores.index.name == "SNPID"
    assert "ID_effect_as_alt" in processed_scores.columns
    assert "ID_effect_as_ref" in processed_scores.columns


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


def test_generate_c_and_t_prs_scores(
    tmp_path, mock_dosage_df, mock_population_allele_frequencies, mock_processed_scores_df
):

    print("mock_dosage_df", mock_dosage_df)
    table = pa.Table.from_pandas(mock_dosage_df)
    test_file = tmp_path / "test_dosage_matrix.feather"
    pa.feather.write_feather(table, test_file)

    with (
        patch(
            "bystro.prs.preprocess_for_prs._extract_af_and_loci_overlap",
            return_value=mock_population_allele_frequencies,
        ),
        patch("bystro.prs.preprocess_for_prs._prune_scores", return_value=mock_processed_scores_df),
    ):

        dosage_matrix_path = test_file
        p_value_threshold = 0.05

        population_vectors = {}
        for population in PopulationVector.__slots__:  # type: ignore
            population_vectors[population] = {"lowerBound": 0.0, "upperBound": 0.0}
        population_vectors["ACB"] = {"lowerBound": 0.9, "upperBound": 0.9}
        population_vectors["CEU"] = {"lowerBound": 0.1, "upperBound": 0.1}

        ancestry_json = {
            "results": [
                {
                    "sampleId": "ID00096",
                    "topHit": {"probability": 0.9, "populations": ["ACB"]},
                    "populations": population_vectors,
                    "superpops": {
                        "AFR": {"lowerBound": 0.9, "upperBound": 0.9},
                        "AMR": {"lowerBound": 0.1, "upperBound": 0.1},
                        "EAS": {"lowerBound": 0.0, "upperBound": 0.0},
                        "EUR": {"lowerBound": 0.0, "upperBound": 0.0},
                        "SAS": {"lowerBound": 0.0, "upperBound": 0.0},
                    },
                    "nSnps": 100,
                },
                {
                    "sampleId": "ID00097",
                    "topHit": {"probability": 0.9, "populations": ["ACB"]},
                    "populations": population_vectors,
                    "superpops": {
                        "AFR": {"lowerBound": 0.9, "upperBound": 0.9},
                        "AMR": {"lowerBound": 0.1, "upperBound": 0.1},
                        "EAS": {"lowerBound": 0.0, "upperBound": 0.0},
                        "EUR": {"lowerBound": 0.0, "upperBound": 0.0},
                        "SAS": {"lowerBound": 0.0, "upperBound": 0.0},
                    },
                    "nSnps": 100,
                },
            ],
            "pcs": {"PC1": [0.1, 0.2], "PC2": [0.3, 0.4]},
        }
        ancestry_json_str = json.dumps(ancestry_json)
        ancestry_results = mjson.decode(ancestry_json_str, type=AncestryResults)

        result = generate_c_and_t_prs_scores(
            assembly="hg19",
            trait="AD",
            pmid="PMID35379992",
            ancestry=ancestry_results,
            dosage_matrix_path=dosage_matrix_path,
            p_value_threshold=p_value_threshold,
            disease_prevalence=0.01,
            continuous_trait=False,
            index_name="foo",
        ).to_dict()
        expected_result = {
            "PRS": {"ID00096": -0.0051128566817909005, "ID00097": -0.0051128566817909005},
            "Corrected OR": {"ID00096": 0.9862663882534203, "ID00097": 0.9862663882534203},
            "Sex": {"ID00096": np.nan, "ID00097": np.nan},
            "Affectation": {"ID00096": np.nan, "ID00097": np.nan},
        }

        assert set(result.keys()) == set(
            expected_result.keys()
        ), "Result keys do not match expected keys"

        assert result["PRS"] == expected_result["PRS"], "PRS values do not match expected values."
        assert (
            result["Corrected OR"] == expected_result["Corrected OR"]
        ), "Corrected OR values do not match expected values."

        for sample in result["Sex"]:
            assert np.isnan(result["Sex"][sample]), f"{sample} Sex status should be nan"

        for sample in result["Affectation"]:
            assert np.isnan(result["Affectation"][sample]), f"{sample} Sex status should be nan"