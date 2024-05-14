import pathlib

import pandas as pd
import pytest

from bystro.proteomics.canopy import Annotations
from bystro.proteomics.canopy.io.annotations.file import read_annotations


@pytest.fixture
def annotation_example_file(tmp_path: pathlib.Path) -> str:
    seq_id = [''] * 8 + ['54321-21', '12345-12']
    soma_id = [''] * 8 + ['SL054321', 'SL012345']
    plasma_scalar = [''] * 8 + ['0.8', '1.1']
    df = pd.DataFrame(
        {
            'SeqId': seq_id,
            'SomaId': soma_id,
            'Plasma Scalar v4.0 to v4.1': plasma_scalar,
        }
    )
    annotations_fp = tmp_path / "annotations.xlsx"
    annotations_fp.touch()

    # Save the DataFrame to an Excel file
    df.to_excel(annotations_fp, index=False)

    yield str(annotations_fp)


def test_read_annotations(annotation_example_file):
    annotations = read_annotations(annotation_example_file)
    assert isinstance(annotations, Annotations)
    assert (annotations.index == ['54321-21', '12345-12']).all()
    assert (annotations.columns == ['Unnamed: 1', 'Unnamed: 2']).all()
