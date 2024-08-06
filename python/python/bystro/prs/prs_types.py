"""Data types for PRS clumping and thresholding method."""

from msgspec import Struct


class PRSJobData(Struct, frozen=True, rename="camel"):
    """Represents the input parameters required to calculate C+T PRS scores."""

    gwas_scores_path: str
    dosage_matrix_path: str
    covariates_path: str | None = None
    map_path: str
    p_value_threshold: float = 0.05
    training_populations: list[str]


class PRSJobResult(Struct, frozen=True):
    """The result of a PRS calculation.

    Represents C+T PRS model output.
    """

    prs_scores: dict[str, float]
