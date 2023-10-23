"""Define the types of inputs and outputs for the PRS module."""

from pathlib import Path
from msgspec import Struct
import pandas as pd

class PRSSubmission(Struct, frozen=True):
    """ The incoming response for a PRS worker.
    Includes the filepath for the target data genotype matrix,
    base GWAS scores, and covariate scores for target data.
    """
    dosage_matrix_path: Path
    association_score_filepath: Path
    covariate_filepath: Path
        
class PRSResponse(Struct, frozen=True):
    """An outgoing response from the PRS worker.
    Represents PRS for an entire experiment as a dataframe.
    """
    dosage_matrix_path: Path
    prs_calculation: pd.DataFrame