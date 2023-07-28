"""Record classes for proteomics module."""

import attrs
from attrs import field


def _tsv_validator(_self: object, _attribute: attrs.Attribute, tsv_filename: str) -> None:
    """Check that tsv_filename ends with ".tsv."""
    if not isinstance(tsv_filename, str):
        err_msg = (
            f"tsv_filename must be of type str, got: {tsv_filename} of type {type(tsv_filename)} instead"
        )
        raise TypeError(err_msg)
    if not tsv_filename.endswith(".tsv"):
        err_msg = f"tsv_filename must be of extension `.tsv`, got {tsv_filename} instead"
        raise ValueError(err_msg)


@attrs.frozen()
class ProteomicsSubmission:
    """Represent an incoming submission to the proteomics worker."""

    tsv_filename: str = field(validator=_tsv_validator)


@attrs.frozen()
class ProteomicsResponse:
    """Represent a proteomics dataframe, converted to json."""

    tsv_filename: str
    dataframe_json: str
