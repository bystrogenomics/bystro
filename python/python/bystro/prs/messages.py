from bystro.beanstalkd.messages import BaseMessage, CompletedJobMessage, SubmittedJobMessage, Struct


class PRSJobData(BaseMessage, frozen=True, forbid_unknown_fields=True, rename="camel"):
    """Data for PRS jobs received from beanstalkd"""

    dosage_matrix_path: str
    output_dir: str
    assembly: str


class PRSJobSubmitMessage(SubmittedJobMessage, frozen=True, kw_only=True, rename="camel"):
    """
    The acknowledgement message to be sent back through beanstalkd
    after PRS job is picked up for processing
    """

    job_config: dict


class PRSJobResult(Struct, frozen=True, forbid_unknown_fields=True, rename="camel"):
    """The contents of the PRS result object to be sent back through beanstalkd"""

    prs_scores_path: str


class PRSJobResultMessage(
    CompletedJobMessage, frozen=True, forbid_unknown_fields=True, kw_only=True, rename="camel"
):
    """
    Results to be sent back through beanstalkd after PRS job is completed
    prs_scores_path: str
        The relative path to the PRS scores file, from the PRSJobData.output_dir
    """

    results: PRSJobResult
