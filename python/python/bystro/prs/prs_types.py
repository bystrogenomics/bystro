"""Define the types of inputs and outputs for the PRS module."""

from pathlib import Path
import pandas as pd

class PRSSubmission:
    """ The incoming response for a PRS worker.
    Includes the filepath for the target data genotype matrix,
    base GWAS scores, and covariate scores for target data.
    """
    def __init__(self, dosage_matrix_path: Path, association_score_filepath: Path, covariate_filepath: Path):
        self.dosage_matrix_path = dosage_matrix_path
        self.association_score_filepath = association_score_filepath
        self.covariate_filepath = covariate_filepath
        
class PRSResponse:
    """An outgoing response from the PRS worker.
    Represents PRS for an entire experiment as a dataframe.
    """
    def __init__(self, dosage_matrix_path: Path, prs_calculation: pd.DataFrame):
        self.dosage_matrix_path = dosage_matrix_path
        self.prs_calculation = prs_calculation