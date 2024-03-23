from os import system
from pathlib import Path
import tempfile

import msgspec
from pyarrow import dataset as ds  # type: ignore

from bystro.utils.compress import get_decompress_to_pipe_cmd
from bystro.ancestry.model import get_models_from_file_system
from bystro.ancestry.inference import AncestryResults, infer_ancestry


# Data is located in bystro/ancestry/data, relative to this script it is ../bystr/ancestry/data
# get that location from this script's path
DATA_DIR = str(Path(__file__).parent.parent / "ancestry" / "data")


def calculate_ancestry_scores(
    vcf: str, assembly: str, dosage: bool = False, out_dir: str | None = None
) -> AncestryResults:
    """Calculate ancestry scores from a VCF file and write the results to a file.

    Args:
        vcf (str): The input VCF file path
        assembly (str): The assembly version (hg19 or hg38)
        out_dir (str, optional): If not provided, the results are not written to a file.
                                 Defaults to None.
        dosage (bool, optional):
            Whether or not to write the dosage matrix output to the output directory.
            If no `out_dir` is not provided, this option cannot be True. Defaults to False.

    Raises:
        RuntimeError: If the bystro-vcf command fails
        ValueError: If `dosage` is True and `out_dir` is not provided

    Returns:
        AncestryResults: The ancestry results
    """
    path_out_dir = None
    if out_dir is not None:
        path_out_dir = Path(out_dir)
        path_out_dir.mkdir(parents=True, exist_ok=True)

    if dosage is None:
        dosage_matrix_path = tempfile.mktemp(suffix=".feather")
    else:
        if path_out_dir is None:
            raise ValueError("If `dosage` is True, `out_dir` must be provided")
        dosage_matrix_path = str(path_out_dir / "dosage_matrix.feather")

    # Input command
    bystro_vcf_command = f"bystro-vcf --noOut --dosageOutput {dosage_matrix_path}"
    cmd = get_decompress_to_pipe_cmd(vcf, bystro_vcf_command)
    res = system(cmd)
    if res != 0:
        raise RuntimeError(f"Failed to run bystro-vcf command: {cmd}")

    # Ancestry command
    dataset = ds.dataset(dosage_matrix_path, format="arrow")

    ancestry_models = get_models_from_file_system(DATA_DIR, assembly)

    results = infer_ancestry(ancestry_models, dataset)

    # Write results to output file
    json_data = msgspec.json.encode(results)

    if path_out_dir is not None:
        with open(str(path_out_dir / "ancestry_results.json"), "wb") as f:
            f.write(json_data)

    return results
