import pandas as pd

ANNOTATION_COLS = ["plex", "channel", "sample", "sample_name", "condition", "replicate"]


def check_df_cols(df: pd.DataFrame, expected_cols: list[str]) -> None:
    actual_cols = df.columns
    if not all(x == y for x, y in zip(expected_cols, actual_cols, strict=False)):
        err_msg = (
            f"expected dataframe to begin with cols: {expected_cols}, got cols: {actual_cols} instead."
        )
        raise ValueError(err_msg)


def prep_annotation_df(annotation_df: pd.DataFrame) -> pd.DataFrame:
    """Prep annotation df, setting index."""
    check_df_cols(annotation_df, ANNOTATION_COLS)
    return annotation_df.set_index("sample")
