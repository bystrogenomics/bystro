from pathlib import Path

from bystro.prs.messages import PRSJobData, PRSJobResult
from bystro.beanstalkd.worker import ProgressPublisher
from bystro.prs.model import get_one_model
from bystro.prs.preprocess_for_prs import generate_c_and_t_prs_scores

def calculate_prs_scores(_publisher: ProgressPublisher, prs_job_data: PRSJobData) -> PRSJobResult:
    """
    Calculate PRS scores for a single submission
    """

    assembly = prs_job_data.assembly
    dosage_matrix_path = prs_job_data.dosage_matrix_path
    p_value_threshold = prs_job_data.p_value_threshold
    disease = prs_job_data.disease
    pmid = prs_job_data.pmid
    
    # TODO 2024-06-16 @akotlar: Update this to get multiple populations, one per individual predicted ancestry
    prs_model = get_one_model(assembly, population="CEU", disease=disease, pmid=pmid)

    result = generate_c_and_t_prs_scores(
        gwas_scores_path = prs_model.score_path,
        dosage_matrix_path = dosage_matrix_path,
        map_path = prs_model.map_path,
        p_value_threshold = p_value_threshold
    )

    print("result", result)
    basename = prs_job_data.out_basename
    out_path = str(Path(prs_job_data.out_dir) / f"{basename}.prs_scores.tsv")

    result.to_csv(out_path, sep="\t")

    return PRSJobResult(result_path=str(out_path), disease=disease, pmid=pmid)
