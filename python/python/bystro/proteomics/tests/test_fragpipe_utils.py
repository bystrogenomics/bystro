import re
import pytest

import pandas as pd

from bystro.proteomics.fragpipe_utils import check_df_cols


def test_check_df_cols_happy_path():
    actual_df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 6, 7]], columns=["a", "b", "c", "d"])
    expected_cols1 = ["a", "b", "c", "d"]
    expected_cols2 = ["a", "b", "c"]
    check_df_cols(actual_df, expected_cols1)
    check_df_cols(actual_df, expected_cols2)


def test_check_df_cols_raises():
    actual_df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 6, 7]], columns=["a", "b", "c", "d"])
    expected_cols = ["a", "b", "z"]
    err_msg = re.escape(
        r"expected dataframe to begin with cols: ['a', 'b', 'z'], "
        r"got cols: Index(['a', 'b', 'c', 'd'], dtype='object') instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        check_df_cols(actual_df, expected_cols)
