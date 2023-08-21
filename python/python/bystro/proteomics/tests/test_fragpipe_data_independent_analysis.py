from io import StringIO
import numpy as np

from bystro.proteomics.fragpipe_data_independent_analysis import DataIndependentAnalysisDataset

raw_pg_matrix_df = pd.DataFrame(
    {
        "Protein.Group": {0: "A0A024RBG1", 1: "A0A075B6H7", 2: "A0A075B6H9"},
        "Protein.Ids": {0: "A0A024RBG1", 1: "A0A075B6H7", 2: "A0A075B6H9"},
        "Protein.Names": {0: np.nan, 1: np.nan, 2: np.nan},
        "Genes": {0: "NUDT4B", 1: "IGKV3-7", 2: "IGLV4-69"},
        "First.Protein.Description": {0: np.nan, 1: np.nan, 2: np.nan},
        "/storage/yihsiao/data/ccrcc_4plexes/CPTAC_CCRCC_W_JHU_20190112_LUMOS_C3N-01522_T.mzML": {
            0: 806691.0,
            1: 38656400.0,
            2: 129411.0,
        },
        "/storage/yihsiao/data/ccrcc_4plexes/CPTAC_CCRCC_W_JHU_20190112_LUMOS_C3N-01179_NAT.mzML": {
            0: 1056910.0,
            1: 74868600.0,
            2: np.nan,
        },
        "/storage/yihsiao/data/ccrcc_4plexes/CPTAC_CCRCC_W_JHU_20190112_LUMOS_C3N-01522_NAT.mzML": {
            0: 1530830.0,
            1: 56854300.0,
            2: np.nan,
        },
        "/storage/yihsiao/data/ccrcc_4plexes/CPTAC_CCRCC_W_JHU_20190112_LUMOS_C3N-01648_NAT.mzML": {
            0: 1337020.0,
            1: 65506700.0,
            2: 478757.0,
        },
    }
)


def test_parse_data_independent_analysis_dataset():
    pg_matrix_handle = StringIO(raw_pg_matrix_df.to_csv(sep="\t"))
