from pathlib import Path
from typing import Any, Callable

from bystro.utils.covariates import ExperimentMappings
from msgspec import json

from bystro.beanstalkd.messages import ProgressPublisher, get_progress_reporter
from bystro.ancestry.ancestry_types import AncestryResults
from bystro.prs.messages import PRSJobData, PRSJobResult
from bystro.prs.preprocess_for_prs import generate_c_and_t_prs_scores

def make_calculate_prs_scores(
    cluster_opensearch_config: dict[str, Any]
) -> Callable[[ProgressPublisher, PRSJobData], PRSJobResult]:
    def calculate_prs_scores(publisher: ProgressPublisher, prs_job_data: PRSJobData) -> PRSJobResult:
        """
        Calculate PRS scores for a single submission
        """

        assembly = prs_job_data.assembly
        dosage_matrix_path = prs_job_data.dosage_matrix_path
        p_value_threshold = prs_job_data.p_value_threshold
        trait = prs_job_data.trait
        pmid = prs_job_data.pmid
        index_name = prs_job_data.index_name
        continuous_trait = prs_job_data.continuous_trait
        disease_prevalence = prs_job_data.disease_prevalence
        training_populations = prs_job_data.training_populations
        distance_based_cluster = prs_job_data.distance_based_cluster
        ld_window_bp = prs_job_data.ld_window_bp
        min_abs_beta = prs_job_data.min_abs_beta
        max_abs_beta = prs_job_data.max_abs_beta


        ancestry: AncestryResults
        with open(prs_job_data.ancestry_result_path, "rb") as f:
            data = f.read()
            ancestry = json.decode(data, type=AncestryResults)

        reporter = get_progress_reporter(publisher, update_interval=100)

        covariates_path = prs_job_data.covariates_path

        covariates_data: ExperimentMappings | None = None
        if covariates_path:
            covariates_data = ExperimentMappings.from_path(covariates_path)

        results = generate_c_and_t_prs_scores(
            assembly=assembly,
            trait=trait,
            pmid=pmid,
            ancestry=ancestry,
            continuous_trait=continuous_trait,
            disease_prevalence=disease_prevalence,
            training_populations=training_populations,
            cluster_opensearch_config=cluster_opensearch_config,
            experiment_mapping=covariates_data,
            index_name=index_name,
            dosage_matrix_path=dosage_matrix_path,
            p_value_threshold=p_value_threshold,
            distance_based_cluster=distance_based_cluster,
            ld_window_bp=ld_window_bp,
            min_abs_beta=min_abs_beta,
            max_abs_beta=max_abs_beta,
            reporter=reporter,
        )

        basename = prs_job_data.out_basename
        out_path = str(Path(prs_job_data.out_dir) / f"{basename}.{trait}.{pmid}.prs.tsv")

        results.to_csv(out_path, sep="\t")

        return PRSJobResult(result_path=str(out_path), trait=trait, pmid=pmid)

    return calculate_prs_scores
