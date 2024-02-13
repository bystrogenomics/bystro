from bystro.prs.messages import PRSJobData, PRSJobResult
from bystro.beanstalkd.worker import ProgressPublisher


def calculate_prs_scores(_publisher: ProgressPublisher, _prs_job_data: PRSJobData) -> PRSJobResult:
    """
    Calculate PRS scores for a single submission
    """

    return PRSJobResult(prs_scores_path="prs_scores_path")
