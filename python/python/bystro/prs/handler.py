from bystro.prs.messages import PRSJobData, PRSJobResult
from bystro.beanstalkd.worker import ProgressPublisher
from bystro.prs.model import PrsModel, get_one_model
from bystro.prs.preprocess_for_prs import generate_c_and_t_prs_scores

import json
from pathlib import Path
def calculate_prs_scores(_publisher: ProgressPublisher, prs_job_data: PRSJobData) -> PRSJobResult:
    """
    Calculate PRS scores for a single submission
    """

    assembly = prs_job_data.assembly
    dosage_matrix_path = prs_job_data.dosage_matrix_path
    p_value_threshold = prs_job_data.p_value_threshold
    
    prs_model = get_one_model(assembly, population="CEU", disease="AD", pmid="PMID35379992")

    result = generate_c_and_t_prs_scores(
        gwas_scores_path = prs_model.score_path,
        dosage_matrix_path = dosage_matrix_path,
        map_path = prs_model.map_path,
        p_value_threshold = p_value_threshold
    )

    print("result", result)

    return PRSJobResult(prs_scores_path="prs_scores_path")
