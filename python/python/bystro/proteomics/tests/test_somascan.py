from pathlib import Path

import pytest

from bystro.proteomics.somascan import (
    ADAT_SAMPLE_ID_COLUMN,
    ADAT_GENE_NAME_COLUMN,
    ADAT_RFU_COLUMN,
    SomascanDataset,
)

adat_path = Path(__file__).parent.parent / "example_data" / "example_data.adat"


def test_somascan_dataset():
    dataset = SomascanDataset.from_paths(str(adat_path))
    assert dataset.adat.shape == (192, 5284)
    assert dataset.annotations is None


def test_passed_bad_arguments():
    valid = SomascanDataset.from_paths(str(adat_path))

    with pytest.raises(
        ValueError,
        match="`adat` argument to SomascanDataset must be a somadata.Adat object",
    ):
        SomascanDataset("foo")

    with pytest.raises(
        ValueError,
        match="`annotations` argument to SomascanDataset must be a somadata.Annotations object",
    ):
        SomascanDataset(valid.adat, "bar")  # type: ignore

    with pytest.raises(FileNotFoundError):
        SomascanDataset.from_paths("foo", "bar")


def test_can_melt_frame():
    dataset = SomascanDataset.from_paths(str(adat_path))

    unique_sample_ids = dataset.adat.index.get_level_values(ADAT_SAMPLE_ID_COLUMN).unique()
    unique_targets = dataset.adat.columns.get_level_values(ADAT_GENE_NAME_COLUMN).unique()

    melted = dataset.to_melted_frame()
    assert melted.shape == (1014528, 55)

    assert melted[ADAT_RFU_COLUMN].dtype == float
    assert melted[ADAT_SAMPLE_ID_COLUMN].dtype == "string[pyarrow_numpy]"
    assert melted[ADAT_GENE_NAME_COLUMN].dtype == "string[pyarrow_numpy]"
    assert set(melted[ADAT_SAMPLE_ID_COLUMN].values) == set(unique_sample_ids)
    assert set(melted[ADAT_GENE_NAME_COLUMN].values) == set(unique_targets)

    columns = melted.columns.tolist()

    assert ADAT_SAMPLE_ID_COLUMN in columns
    assert ADAT_GENE_NAME_COLUMN in columns
    assert ADAT_RFU_COLUMN in columns
