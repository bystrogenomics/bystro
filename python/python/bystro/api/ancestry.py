import csv
import json
from os import system
from pathlib import Path
import tempfile

import msgspec
from pyarrow import dataset as ds  # type: ignore

from bystro.utils.compress import get_decompress_to_pipe_cmd
from bystro.ancestry.model import get_models
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

    ancestry_models = get_models(assembly)

    results = infer_ancestry(ancestry_models, dataset)

    # Write results to output file
    json_data = msgspec.json.encode(results)

    if path_out_dir is not None:
        with open(str(path_out_dir / "ancestry_results.json"), "w") as f:
            f.write(str(json_data, "utf-8"))

    return results


def ancestry_json_to_format(input_json_path, output_path, output_format="tsv"):
    """
    Parse the JSON output from the Ancestry Inference tool and write it to a TSV or CSV file.

    Arguments
    ---------
        input_json_path (str): Path to the JSON file to parse.
        output_tsv_path (str): Path to the output TSV or CSV file.

    Returns
    -------
    None
    """
    if output_format != "tsv" and output_format != "csv":
        raise ValueError("output_format must be either 'tsv' or 'csv'")

    with open(input_json_path, "r") as json_file:
        data = json.load(json_file)

    results = data["results"]

    delimiter = "\t" if output_format == "tsv" else ","
    # Make output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open TSV file for writing
    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=delimiter)

        # Write header
        header = [
            "sample_id",
            "top_population",
            "top_population_probability",
            "top_superpopulation",
            "top_superpopulation_probability",
        ]
        populations = list(results[0]["populations"].keys())
        superpops = list(results[0]["superpops"].keys())

        header += populations + superpops
        writer.writerow(header)

        # Write data rows
        for result in results:
            row = [
                result["sampleId"],
                ";".join(result["topHit"]["populations"]),
                result["topHit"]["probability"],
            ]

            top_hit_superpop = ""
            top_hit_superpop_probability = 0
            for superpop, vals in result["superpops"].items():
                mean = (vals["lowerBound"] + vals["upperBound"]) / 2

                if mean > top_hit_superpop_probability:
                    top_hit_superpop_probability = mean
                    top_hit_superpop = superpop

            row.append(top_hit_superpop)
            row.append(top_hit_superpop_probability)

            for population in populations:
                row.append(
                    (
                        result["populations"][population]["lowerBound"]
                        + result["populations"][population]["upperBound"]
                    )
                    / 2
                )

            for superpop in superpops:
                row.append(
                    (
                        result["superpops"][superpop]["lowerBound"]
                        + result["superpops"][superpop]["upperBound"]
                    )
                    / 2
                )

            writer.writerow(row)
