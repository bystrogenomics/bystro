from bystro.proteomics.proteomics_types import (
    ProteomicsSubmission,
    DataFrameJson,
    ProteomicsResponse,
)
import pandas as pd
import numpy as np

pd.options.future.infer_string = True  # type: ignore


def test_ProteomicsSubmission():
    ProteomicsSubmission("foo.tsv")


def test_ProteomicsResonse():
    data = pd.DataFrame(
        np.random.random((2, 3)), index=["sample1", "sample2"], columns=["gene1", "gene2", "gene3"]
    )
    json_dataframe = DataFrameJson.from_df(data)
    ProteomicsResponse("foo.tsv", json_dataframe)
