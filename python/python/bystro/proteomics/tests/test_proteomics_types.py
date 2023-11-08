import pytest
from bystro.proteomics.proteomics_types import (
    ProteomicsSubmission,
    DataFrameJson,
    ProteomicsResponse,
)
import pandas as pd
import numpy as np


def test_ProteomicsSubmission():
    ProteomicsSubmission("foo.tsv")


def test_ProteomicsSubmission_bad_input():
    with pytest.raises(TypeError, match="must be of type str"):
        ProteomicsSubmission(3) # type: ignore

    with pytest.raises(ValueError, match="must be of extension `.tsv`"):
        ProteomicsSubmission("foo.docx")


def test_ProteomicsResonse():
    data = pd.DataFrame(
        np.random.random((2, 3)), index=["sample1", "sample2"], columns=["gene1", "gene2", "gene3"]
    )
    json_dataframe = DataFrameJson.from_df(data)
    ProteomicsResponse("foo.tsv", json_dataframe)
