from pathlib import Path
from typing import Any, Callable

from msgspec import json

from bystro.beanstalkd.worker import ProgressPublisher
from bystro.ancestry.ancestry_types import AncestryResults
from bystro.prs.messages import PRSJobData, PRSJobResult
from bystro.prs.preprocess_for_prs import generate_c_and_t_prs_scores


def make_calculate_prs_scores(
    cluster_opensearch_config: dict[str, Any]
) -> Callable[[ProgressPublisher, PRSJobData], PRSJobResult]:
    def calculate_prs_scores(_publisher: ProgressPublisher, prs_job_data: PRSJobData) -> PRSJobResult:
        """
        Calculate PRS scores for a single submission
        """

        assembly = prs_job_data.assembly
        dosage_matrix_path = prs_job_data.dosage_matrix_path
        p_value_threshold = prs_job_data.p_value_threshold
        disease = prs_job_data.disease
        pmid = prs_job_data.pmid
        index_name = prs_job_data.index_name

        ancestry: AncestryResults
        with open(prs_job_data.ancestry_result_path, "rb") as f:
            data = f.read()
            ancestry = json.decode(data, type=AncestryResults)

        result = generate_c_and_t_prs_scores(
            assembly=assembly,
            disease=disease,
            pmid=pmid,
            ancestry=ancestry,
            cluster_opensearch_config=cluster_opensearch_config,
            index_name=index_name,
            dosage_matrix_path=dosage_matrix_path,
            p_value_threshold=p_value_threshold,
        )

        basename = prs_job_data.out_basename
        out_path = str(Path(prs_job_data.out_dir) / f"{basename}.prs_scores.tsv")

        result.to_csv(out_path, sep="\t")

        return PRSJobResult(result_path=str(out_path), disease=disease, pmid=pmid)

    return calculate_prs_scores
