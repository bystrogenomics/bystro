"""Classify genotypes at inference time."""

import logging
import os
import psutil

from msgspec import Struct
import numpy as np
import pandas as pd

import pyarrow.compute as pc  # type: ignore
from pyarrow.dataset import Dataset  # type: ignore

from sklearn.ensemble import RandomForestClassifier

from bystro.ancestry.ancestry_types import (
    AncestryResults,
    AncestryScoresOneSample,
    AncestryTopHit,
    PopulationVector,
    ProbabilityInterval,
    SuperpopVector,
)
from bystro.ancestry.asserts import assert_equals
from bystro.ancestry.train import POPS, SUPERPOP_FROM_POP, SUPERPOPS
from bystro.utils.timer import Timer

logger = logging.getLogger(__name__)


class AncestryModel(Struct, frozen=True, forbid_unknown_fields=True, rename="camel"):
    """Bundle together PCA and RFC models for bookkeeping purposes."""

    pca_loadings_df: pd.DataFrame
    rfc: RandomForestClassifier

    def __post_init__(self) -> None:
        """Ensure that PCA and RFC features line up correctly."""
        pca_cols = self.pca_loadings_df.columns
        rfc_features = self.rfc.feature_names_in_
        if not (len(pca_cols) == len(rfc_features) and (pca_cols == rfc_features).all()):
            err_msg = (
                f"PC loadings columns:{self.pca_loadings_df.columns} must equal "
                f"rfc.feature_names_in: {self.rfc.feature_names_in_}"
            )
            raise ValueError(err_msg)

    def predict_proba(self, genotypes: pd.DataFrame) -> tuple[dict[str, list[float]], pd.DataFrame]:
        """
        Predict population probabilities from dosage matrix.

        Args:
            genotypes: pd.DataFrame, shape (n_variants, m_samples)
                A dosage matrix with samples as rows and variants as columns.
        """
        logger.debug("computing PCA transformation")

        with Timer() as timer:
            Xpc = genotypes.T @ self.pca_loadings_df

        logger.debug("finished computing PCA transformation in %f seconds", timer.elapsed_time)
        logger.debug("computing RFC classification")

        with Timer() as timer:
            probs = self.rfc.predict_proba(Xpc)

        logger.debug("finished computing RFC classification in %f seconds", timer.elapsed_time)

        Xpc_dict = Xpc.T.to_dict(orient="list")

        return Xpc_dict, pd.DataFrame(probs, index=genotypes.columns, columns=POPS)


def _package_ancestry_response_from_pop_probs(
    pcs_for_plotting: dict[str, list[float]],
    pop_probs_df: pd.DataFrame,
    missingnesses: pd.Series,
) -> AncestryResults:
    """Fill out AncestryResults using filepath, numerical model output and sample-wise missingnesses."""
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
        max_value = float(max(pop_probs_dict.values()))
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
            AncestryScoresOneSample(
                sample_id=sample_id,
                top_hit=AncestryTopHit(probability=max_value, populations=top_pops),
                populations=pop_vector,
                superpops=superpop_vector,
                missingness=float(missingnesses[sample_id]),
            )
        )

    return AncestryResults(results=ancestry_results, pcs=pcs_for_plotting)


def infer_ancestry(ancestry_model: AncestryModel, genotypes: Dataset) -> AncestryResults:
    """
    Infer ancestry from genotypes using a trained model.

    Parameters
    ----------
    ancestry_model: AncestryModel
        A trained model for predicting ancestry from genotypes.

    genotypes: Arrow Dataset, shape (n_variants, m_samples)
        A dataset containing genotypes to be classified.

    Returns
    -------
    AncestryResults
        A Struct of ancestry results for each sample in the dataset.
        AncestryResults.results is a list of AncestryScoresOneSample objects and
        AncestryResults.pcs is a dictionary of principal components for each sample.

        AncestryScoresOneSample is a Struct with the following fields:
        - sample_id: str
            The ID of the sample. This is the same as the sample ID in the input dataset.
        - top_hit: AncestryTopHit
            The top hit for a sample, with the max value (a probability) and the list of population(s)
            corresponding to that probability, typically a single population.
        - populations: PopulationVector
            A Struct of population probabilities for the sample. For instance, if the sample is
            80% Finnish and 20% Nigerian, the PopulationVector
            would be {"FIN": 0.8, "YRI": 0.2}.
        - superpops: SuperpopVector
            A Struct of super population probabilities for the sample. For instance, if the sample is
            80% European and 20% African, the PopulationVector would be {"AFR": 0.2, "EUR": 0.8}.
        - missingness: float
            The fraction of expected variants (those in the AncestryModel)
            found missing in the sample.

            Note that this is not the overall sample missingness but rather
            the sample missingness relative to the AncestryModel.
    """

    logger.debug("Beginning ancestry inference")

    with Timer() as timer:
        mask = pc.field("locus").isin(ancestry_model.pca_loadings_df.index)
        scanner = genotypes.scanner(filter=mask)

        # Initialize an empty DataFrame or a list to collect batches
        filtered_genotypes = []

        for batch in scanner.to_batches():
            # Convert the current batch to a pandas DataFrame
            batch_df = batch.to_pandas()

            # Set the index to 'locus', assuming 'locus' is a column in your dataset
            batch_df = batch_df.set_index("locus")

            # Append the processed batch to the list
            filtered_genotypes.append(batch_df)

        # Concatenate all batches into a single DataFrame
        genotypes_df = pd.concat(filtered_genotypes)

    logger.info(
        "Memory usage after dosage matrix filtering: %s (MB)",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    )

    logger.info("Completed dosage matrix filtering in %f seconds", timer.elapsed_time)

    with Timer() as timer:
        # TODO: @akotlar 2024-01-31: Replace reliance on imputation with Austin Talbot's model
        # which is robust to missing data.
        missing_rows = list(set(ancestry_model.pca_loadings_df.index) - set(genotypes_df.index))
        if missing_rows:
            missing_rows_df = pd.DataFrame(
                np.ones((len(missing_rows), len(genotypes_df.columns))),
                index=missing_rows,
                columns=genotypes_df.columns,
            )
            genotypes_df = pd.concat([genotypes_df, missing_rows_df])

        missingness = len(missing_rows) / len(ancestry_model.pca_loadings_df.index)
        sample_missingnesses = genotypes_df.isna().mean(axis=0) + missingness

        genotypes_df[genotypes_df.isna()] = 1

        pcs_for_plotting, pop_probs_df = ancestry_model.predict_proba(genotypes_df)

    logger.info("Completed ancestry inference in %f seconds", timer.elapsed_time)

    return _package_ancestry_response_from_pop_probs(
        pcs_for_plotting, pop_probs_df, sample_missingnesses
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
    # The ancestry calculations come out as np.float64, which is not JSON serializable in msgspec.
    return ProbabilityInterval(float(x), float(x))
