from pathlib import Path

import pytest

from bystro.proteomics.somascan import SomascanDataset

adat_path = Path(__file__).parent.parent / "example_data" / "example_data.adat"


def test_somascan_dataset():
    dataset = SomascanDataset.from_paths(str(adat_path))
    assert dataset.adat.shape == (192, 5284)
    assert dataset.annotations is None


def test_passed_bad_arguments():
    valid = SomascanDataset.from_paths(str(adat_path))

    with pytest.raises(
        ValueError,
        match="`adat` argument to SomascanDataset must be a canopy.Adat object",
    ):
        SomascanDataset("foo")

    with pytest.raises(
        ValueError,
        match="`annotations` argument to SomascanDataset must be a canopy.Annotations object",
    ):
        SomascanDataset(valid.adat, "bar")  # type: ignore

    with pytest.raises(FileNotFoundError):
        SomascanDataset.from_paths("foo", "bar")
