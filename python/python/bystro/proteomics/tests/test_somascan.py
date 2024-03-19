from pathlib import Path
from bystro.proteomics.somascan import SomascanDataset


def test_somascan_dataset():
    adat_path = Path(__file__).parent.parent / "example_data" / "example_data.adat"

    dataset = SomascanDataset.from_paths(str(adat_path))
    assert dataset.adat.shape == (192, 5284)
    assert dataset.annotations is None