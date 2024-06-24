"""Record classes for proteomics module."""

import pandas as pd
from msgspec import Struct, DecodeError
import msgspec.json as mjson

pd.options.future.infer_string = True  # type: ignore

decoder = mjson.Decoder()


# The motivation for this class is that we only want to instantiate by creating from a correctly
# json-serialized pd.DataFrame, ensuring we have a string-like class that can't represent anything
# but a json-ified df.


class DataFrameJson(Struct, frozen=True):
    """Represent a DataFrame as a JSON string."""

    json_payload: str

    def __post_init__(self) -> None:
        try:
            decoder.decode(self.json_payload)
        except DecodeError as e:
            raise ValueError("Invalid JSON payload") from e

    @classmethod
    def from_df(cls, dataframe: pd.DataFrame) -> "DataFrameJson":
        """Convert a pd.DataFrame to JsonDataFrame"""
        return cls(dataframe.to_json(orient="table"))

    def to_df(self) -> pd.DataFrame:
        """Read out JsonDataFrame to pd.DataFrame"""
        return pd.read_json(self.json_payload, orient="table")


class ProteomicsSubmission(Struct, frozen=True):
    """Represent an incoming submission to the proteomics worker."""

    tsv_filename: str


class ProteomicsResponse(Struct, frozen=True):
    """Represent a proteomics dataframe, converted to json."""

    tsv_filename: str
    dataframe_json: DataFrameJson
