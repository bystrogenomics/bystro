import pickle
import os

import pandas as pd
from pandas.testing import assert_frame_equal

from bystro.parent_of_origin.poirot import extract_residuals, do_poirot_by_snp

pd.options.future.infer_string = True  # type: ignore

def get_data_path():
    # get data from script path
    script_path = os.path.dirname(os.path.realpath(__file__))
    return f"{script_path}/poirot_data.pkl"

def test_do_poirot_by_snp():
    with open(get_data_path(), "rb") as f:
        loaded_data_dict = pickle.load(f)
    PHENO = loaded_data_dict["phenotypes"]
    GENO = loaded_data_dict["variants"]
    COVAR = loaded_data_dict["covariates"]

    PHENO_ADJ = extract_residuals(PHENO, COVAR)

    results = pd.DataFrame([do_poirot_by_snp(i, PHENO_ADJ, GENO) for i in range(GENO.shape[1])])
    results["variant"] = GENO.columns.astype("string[pyarrow_numpy]")

    expected_data = {
        "pval": [0.672187, 0.397630],
        "stat": [0.805646, 1.052683],
        "variant": ["snp1", "snp2"],
    }
    expected = pd.DataFrame(expected_data)

    assert_frame_equal(results, expected)
