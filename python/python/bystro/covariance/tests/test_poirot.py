import pandas as pd
from pandas.testing import assert_frame_equal
import pickle
from bystro.covariance.poirot import extract_residuals, do_poirot_by_snp


def test_do_poirot_by_snp():
    with open("poirot_data.pkl", "rb") as f:
        loaded_data_dict = pickle.load(f)
    PHENO = loaded_data_dict["phenotypes"]
    GENO = loaded_data_dict["variants"]
    COVAR = loaded_data_dict["covariates"]

    PHENO_ADJ = extract_residuals(PHENO, COVAR)

    results = pd.DataFrame(
        [do_poirot_by_snp(i, PHENO_ADJ, GENO) for i in range(GENO.shape[1])]
    )
    results["variant"] = GENO.columns

    expected_data = {
        "pval": [0.672187, 0.397630],
        "stat": [0.805646, 1.052683],
        "variant": ["snp1", "snp2"],
    }
    expected = pd.DataFrame(expected_data)

    assert_frame_equal(results, expected)
