from bystro.beanstalkd.messages import BaseMessage, CompletedJobMessage, SubmittedJobMessage, Struct


class PRSJobData(BaseMessage, frozen=True, forbid_unknown_fields=True, rename="camel"):
    """
    The message to be sent through beanstalkd to initiate a PRS job

    Attributes:
    dosage_matrix_path: str
        The path to the dosage matrix file
    out_dir: str
        The directory to save the output file
    out_basename: str
        The basename of the output file
    assembly: str
        The assembly to use for the PRS calculation
    trait: str
        The trait, disease, or outcome to use for the PRS calculation
        (e.g Alzheimer's Disease, ADHD, Schizophrenia)
    pmid: str
        The ID of the study (PubMed ID typically)
    p_value_threshold: float
        The p-value threshold to use for the PRS calculation
    ancestry_result_path: str, optional
        The path to the ancestry result file. Must be provided if populations_path is not provided.

        When populations_path is not provided, the inferred ancestry
        from the ancestry_result_path will be used to calculate best-fit populations,
        which are used for allele frequency weighting in the PRS calculation.
    populations_path: str, optional
        The path to the populations file, which is expected to be 2 tab-separated file with 2 columns,
        the first being the sample name, and the 2nd being the population.

        Must be provided if ancestry_result_path is not provided.

        When both ancestry_result_path and population_path are provided,
        the populations in this file will be used for allele frequency weighting in the PRS calculation.
    index_name: str | None
        The index of the dataset in the OpenSearch cluster, if available
    covariates_path: str | None
        The path to the covariates file, which is expected to be a tab-separated file with the first column
        being the sample id, and the subsequent columns being the covariates.

        Bystro understands the following reserved covariates:
            1. "phenotype" - The sample phenotype. Expected to be 0 or 1.
    """

    dosage_matrix_path: str
    out_dir: str
    out_basename: str
    assembly: str
    trait: str
    pmid: str
    p_value_threshold: float
    ancestry_result_path: str
    disease_prevalence: float
    binary_trait: bool
    populations_path: str | None = None
    index_name: str | None = None
    covariates_path: str | None = None


class PRSJobSubmitMessage(SubmittedJobMessage, frozen=True, kw_only=True, rename="camel"):
    """
    The acknowledgement message to be sent back through beanstalkd
    after PRS job is picked up for processing
    """

    job_config: dict


class PRSJobResult(Struct, frozen=True, forbid_unknown_fields=True, rename="camel"):
    """The contents of the PRS result object to be sent back through beanstalkd"""

    result_path: str
    trait: str
    pmid: str


class PRSJobResultMessage(
    CompletedJobMessage, frozen=True, forbid_unknown_fields=True, kw_only=True, rename="camel"
):
    """
    Results to be sent back through beanstalkd after PRS job is completed
    result_path: str
        The relative path to the PRS scores file, from the PRSJobData.output_dir
    """

    results: PRSJobResult