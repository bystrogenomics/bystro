"""Classify genotypes at inference time."""
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from bystro.ancestry.ancestry_types import (
    AncestryResponse,
    AncestryResult,
    PopulationVector,
    ProbabilityInterval,
    SuperpopVector,
)
from bystro.ancestry.asserts import assert_equals
from bystro.ancestry.train import POPS, SUPERPOP_FROM_POP, SUPERPOPS
from bystro.utils.timer import Timer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AncestryModel:
    """Bundle together PCA and RFC models for bookkeeping purposes."""

    pca_loadings_df: pd.DataFrame
    rfc: RandomForestClassifier

    def __post_init__(self) -> "AncestryModel":
        """Ensure that PCA and RFC features line up correctly."""
        pca_cols = self.pca_loadings_df.columns
        rfc_features = self.rfc.feature_names_in_
        if not (len(pca_cols) == len(rfc_features) and (pca_cols == rfc_features).all()):
            err_msg = (
                f"PC loadings columns:{self.pca_loadings_df.columns} must equal "
                f"rfc.feature_names_in: {self.rfc.feature_names_in_}"
            )
            raise ValueError(err_msg)
        return self

    def predict_proba(self, genotypes: pd.DataFrame) -> tuple[dict[str, list[float]], pd.DataFrame]:
        """Predict population probabilities from dosage matrix."""
        logger.debug("computing PCA transformation")
        with Timer() as timer:
            Xpc = genotypes @ self.pca_loadings_df
        logger.debug("finished computing PCA transformation in %f seconds", timer.elapsed_time)
        logger.debug("computing RFC classification")
        with Timer() as timer:
            probs = self.rfc.predict_proba(Xpc)
        logger.debug("finished computing RFC classification in %f seconds", timer.elapsed_time)
        Xpc_dict = Xpc.T.to_dict(orient="list")
        return Xpc_dict, pd.DataFrame(probs, index=genotypes.index, columns=POPS)


def _fill_missing_data(genotypes: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    sample_missingnesses = genotypes.isna().mean(axis="columns")  # average over columns, leaving indices
    # todo: much better imputation strategy to come, but we're stubbing out for now.
    # if col completely missing in all samples, just fill as heterozygote for now
    mean_column_values = genotypes.mean(axis="index").fillna(1)
    imputed_genotypes = genotypes.fillna(mean_column_values)
    return imputed_genotypes, sample_missingnesses


def _package_ancestry_response_from_pop_probs(
    vcf_path: Path | str,
    pcs_for_plotting: dict[str, list[float]],
    pop_probs_df: pd.DataFrame,
    missingnesses: pd.Series,
) -> AncestryResponse:
    """Fill out AncestryResponse using filepath, numerical model output and sample-wise missingnesses."""
    superpop_probs_df = _superpop_probs_from_pop_probs(pop_probs_df)
    ancestry_results = []

    for (sample_id, sample_pop_probs), (_sample_id2, sample_superpop_probs) in zip(
        pop_probs_df.iterrows(), superpop_probs_df.iterrows(), strict=True
    ):
        if not isinstance(sample_id, str):
            # just spoonfeeding mypy here-- this should never raise
            err_msg = (
                f"Expected sample_id of type str, got {sample_id} of type({type(sample_id)}) instead"
            )
            raise TypeError(err_msg)

        pop_probs_dict = dict(sample_pop_probs)
        max_value = max(pop_probs_dict.values())
        top_pops = [pop for pop, value in pop_probs_dict.items() if value == max_value]

        pop_vector = PopulationVector(
            **{
                pop: _make_trivial_probability_interval(value)
                for (pop, value) in dict(sample_pop_probs).items()
            }
        )
        superpop_vector = SuperpopVector(
            **{
                superpop: _make_trivial_probability_interval(value)
                for (superpop, value) in dict(sample_superpop_probs).items()
            }
        )
        ancestry_results.append(
            AncestryResult(
                sample_id=sample_id,
                top_hit=(max_value, top_pops),
                populations=pop_vector,
                superpops=superpop_vector,
                missingness=missingnesses[sample_id],
            )
        )
    return AncestryResponse(vcf_path=str(vcf_path), results=ancestry_results, pcs=pcs_for_plotting)


# TODO: implement with ray
def infer_ancestry(
    ancestry_model: AncestryModel, genotypes: pd.DataFrame, vcf_path: Path | str
) -> AncestryResponse:
    """Run an ancestry job."""
    # TODO: main ancestry model logic goes here.  Just stubbing out for now.

    logger.debug("Filling missing data for VCF: %s", vcf_path)
    with Timer() as timer:
        imputed_genotypes, missingnesses = _fill_missing_data(genotypes)
    logger.debug("Finished filling missing data for VCF in %f seconds", timer.elapsed_time)
    pcs_for_plotting, pop_probs_df = ancestry_model.predict_proba(imputed_genotypes)
    return _package_ancestry_response_from_pop_probs(
        vcf_path, pcs_for_plotting, pop_probs_df, missingnesses
    )


def _superpop_probs_from_pop_probs(pop_probs: pd.DataFrame) -> pd.DataFrame:
    """Given a matrix of population probabilities, convert to matrix of superpop probabilities."""
    N = len(pop_probs)
    pops = sorted(SUPERPOP_FROM_POP.keys())
    superpops = sorted(set(SUPERPOP_FROM_POP.values()))
    superpop_projection_matrix = pd.DataFrame(
        np.array([[int(superpop == SUPERPOP_FROM_POP[pop]) for superpop in superpops] for pop in pops]),
        index=POPS,
        columns=SUPERPOPS,
    )
    superpop_probs = pop_probs @ superpop_projection_matrix
    assert_equals(
        "Expected superpop_probs shape (N x |superpops|):",
        superpop_probs.shape,
        "Actual shape",
        (N, len(superpops)),
    )
    return superpop_probs


def _make_trivial_probability_interval(x: float) -> ProbabilityInterval:
    """Promote a value to a trivial ProbabilityInterval with equal lower, upper bounds."""
    return ProbabilityInterval(x, x)
