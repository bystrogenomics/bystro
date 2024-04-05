from os import system
from pathlib import Path
import tempfile

import msgspec
from pyarrow import dataset as ds  # type: ignore

from bystro.utils.compress import get_decompress_to_pipe_cmd
from bystro.ancestry.model import get_models_from_file_system
from bystro.ancestry.inference import AncestryResults, infer_ancestry


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

    if not dosage:
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

    return calculate_ancestry_scores_from_dosage(dosage_matrix_path, assembly, out_dir)


def calculate_ancestry_scores_from_dosage(
    dosage_matrix_path: str, assembly: str, out_dir: str | None = None
) -> AncestryResults:
    """Calculate ancestry scores from a Bystro dosage Arrow feather file and write the results.

    Args:
        dosage_matrix_path (str): The input VCF file path
        assembly (str): The assembly version (hg19 or hg38)
        out_dir (str, optional): If not provided, the results are not written to a file.
                                 Defaults to None.

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

    # Ancestry command
    dataset = ds.dataset(dosage_matrix_path, format="arrow")

    ancestry_models = get_models_from_file_system(assembly)

    results = infer_ancestry(ancestry_models, dataset)

    # Write results to output file
    json_data = msgspec.json.encode(results)

    if path_out_dir is not None:
        with open(str(path_out_dir / "ancestry_results.json"), "w") as f:
            f.write(str(json_data, "utf-8"))

    return results
