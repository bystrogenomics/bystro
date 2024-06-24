"""Classify genotypes at inference time."""

import logging
import gc
import os
import psutil
import warnings

from msgspec import Struct
import numpy as np
import pandas as pd

import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
from pyarrow.dataset import Dataset  # type: ignore

from sklearn.ensemble import RandomForestClassifier  # type: ignore

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
warnings.simplefilter(action="ignore", category=FutureWarning)

pd.options.future.infer_string = True  # type: ignore

ANCESTRY_SCORE_SAMPLE_CHUNK_SIZE = int(os.getenv("ANCESTRY_SCORE_SAMPLE_CHUNK_SIZE", 200))


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
    n_snps: int,
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
                n_snps=n_snps,
            )
        )

    return AncestryResults(results=ancestry_results, pcs=pcs_for_plotting)


class AncestryModels(Struct, frozen=True, forbid_unknown_fields=True, rename="camel"):
    """
    A Struct of trained models for predicting ancestry,
    using either gnomAD PC's or genotyping array PC's, depending on which has lower missingness.
    """

    gnomad_model: AncestryModel
    array_model: AncestryModel


def infer_ancestry(ancestry_models: AncestryModels, genotypes: Dataset) -> AncestryResults:
    """
    Infer ancestry from genotypes using a trained model.

    Parameters
    ----------
    ancestry_models: AncestryModels
        A Struct of trained models for predicting ancestry,

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
        - num_snps_selected: int
            The number of SNPs used to infer ancestry for the sample.
    """

    logger.debug("Beginning ancestry inference")

    logger.info(
        "Memory usage before dosage matrix filtered row counting: %s (MB)",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    )

    pool = pa.default_memory_pool()

    with Timer() as timer:
        gnomad_model = ancestry_models.gnomad_model
        array_model = ancestry_models.array_model

        mask_gnomad = pc.field("locus").isin(gnomad_model.pca_loadings_df.index)
        mask_array = pc.field("locus").isin(array_model.pca_loadings_df.index)

        scanner_gnomad = genotypes.filter(mask_gnomad)
        scanner_array = genotypes.filter(mask_array)

        gnomad_matching_row_count = scanner_gnomad.count_rows(memory_pool=pool)
        array_matching_row_count = scanner_array.count_rows(memory_pool=pool)

        logger.debug(
            "Found %d rows in genotypes matching gnomAD PCA loadings",
            gnomad_matching_row_count,
        )
        logger.debug(
            "Found %d rows in genotypes matching array PCA loadings",
            array_matching_row_count,
        )

        logger.info(
            "Memory usage after dosage matrix filtered row counting: %s (MB)",
            psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        )

        num_snps_selected = max(gnomad_matching_row_count, array_matching_row_count)
        if gnomad_matching_row_count >= array_matching_row_count:
            scanner = scanner_gnomad
            ancestry_model = gnomad_model

            logger.debug("Using gnomAD PCA loadings for ancestry inference due to lower missingness")

            del scanner_array
            del array_model
        else:
            scanner = scanner_array
            ancestry_model = array_model

            logger.debug("Using array PCA loadings for ancestry inference due to lower missingness")

            del scanner_gnomad
            del gnomad_model

    logger.info("Completed ancestry model selection in %f seconds", timer.elapsed_time)

    samples = [name for name in genotypes.schema.names if name != "locus"]

    # Take chunks of up to 500 samples
    chunk_size = ANCESTRY_SCORE_SAMPLE_CHUNK_SIZE
    num_samples = len(samples)
    start = 0

    all_pcs_for_plotting = {}
    all_pop_probs_df = []
    while start < num_samples:
        with Timer() as timer:
            end = min(start + chunk_size, num_samples)
            chunk_samples = samples[start:end]

            genotypes_chunk_table = scanner.to_table(["locus", *chunk_samples], memory_pool=pool)

            logger.info(
                "Memory usage after dosage matrix filtering for samples %d to %d: %s (MB)",
                start,
                end,
                psutil.Process(os.getpid()).memory_info().rss / 1024**2,
            )

            genotypes_df = genotypes_chunk_table.to_pandas().set_index("locus")

            # If there are duplicates, log them and remove them
            if genotypes_df.index.duplicated().any():
                logger.warning("Found duplicate loci in genotypes, removing duplicates")
                genotypes_df = genotypes_df[~genotypes_df.index.duplicated(keep="first")]

            logger.info(
                "Memory usage after converting table to Pandas dataframe, for samples %d to %d: %s (MB)",
                start,
                end,
                psutil.Process(os.getpid()).memory_info().rss / 1024**2,
            )

            missing_rows = list(set(ancestry_model.pca_loadings_df.index) - set(genotypes_df.index))
            if missing_rows:
                missing_rows_df = pd.DataFrame(
                    np.ones((len(missing_rows), len(genotypes_df.columns))),
                    index=missing_rows,
                    columns=genotypes_df.columns,
                )
                genotypes_df = pd.concat([genotypes_df, missing_rows_df])

            genotypes_df[genotypes_df.isna() | (genotypes_df < 0)] = 1

            pcs_for_plotting, pop_probs_df = ancestry_model.predict_proba(genotypes_df)

            all_pcs_for_plotting.update(pcs_for_plotting)
            all_pop_probs_df.append(pop_probs_df)

        # We must manually free memory, because
        # otherwise it appears Python has a tough time coping
        # with the size of allocations, which can reach into gigabytes per loop iteration
        del genotypes_df
        del genotypes_chunk_table
        pool.release_unused()
        gc.collect()

        logger.info(
            "Completed ancestry inference for samples %d to %d in %f seconds. RSS: %s (MB)",
            start,
            end,
            timer.elapsed_time,
            psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        )

        start = end

    return _package_ancestry_response_from_pop_probs(
        all_pcs_for_plotting, pd.concat(all_pop_probs_df), num_snps_selected
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
