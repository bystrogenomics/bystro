import re
import pytest

import pandas as pd

from bystro.proteomics.fragpipe_utils import check_df_starts_with_cols

pd.options.future.infer_string = True  # type: ignore


def test_check_df_starts_with_cols_happy_path():
    """Ensure that check_df_starts_with_cols succeeds if df cols start with given cols."""
    actual_df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 6, 7]], columns=["a", "b", "c", "d"])
    expected_cols1 = ["a", "b", "c", "d"]
    expected_cols2 = ["a", "b", "c"]
    check_df_starts_with_cols(actual_df, expected_cols1)
    check_df_starts_with_cols(actual_df, expected_cols2)


def test_check_df_cols_raises():
    """Ensure that check_df_starts_with_cols raises if df cols don't start with given cols."""
    actual_df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 6, 7]], columns=["a", "b", "c", "d"])
    expected_cols = ["a", "b", "z"]
    err_msg = re.escape(
        r"expected dataframe to begin with cols: ['a', 'b', 'z'], "
        r"got cols: Index(['a', 'b', 'c', 'd'], dtype='string') instead."
    )
    with pytest.raises(ValueError, match=err_msg):
        check_df_starts_with_cols(actual_df, expected_cols)
