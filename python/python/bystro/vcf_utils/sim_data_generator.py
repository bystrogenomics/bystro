"""VCF index and data simulator for testing analysis modules."""

import random
from io import StringIO

import pandas as pd

HEADER_COLS = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]


def generate_random_vcf_index() -> tuple[int, int, str, str]:
    """Generate autosomal chr vcf variant IDs to use for testing."""
    random_chr = random.randint(1, 22+1)
    random_pos = random.randint(1, 1000000)
    random_ref = random.choice(["A", "T", "C", "G"])
    random_alt = random.choice([letter for letter in ["A", "T", "C", "G"] if letter != random_ref])
    return random_chr,random_pos,random_ref,random_alt


def generate_simulated_vcf(
    num_samples:int, num_vars:int
) -> tuple[pd.DataFrame, str]:
    """VCF simulator for testing analysis modules."""
    # This is a first pass - could include more sophisticated choices for possible values in future
    sample_ids = [f"SampleID{i+1}" for i in range(num_samples)]
    header = HEADER_COLS + sample_ids
    vcf_data = []
    simulated_indices = []
    vcf_data.append("\t".join(header))
    for position in range(num_vars):
        chrom = "chr1"
        pos = position
        variant_id = generate_random_vcf_index()  # Using simulated index as variant ID
        simulated_indices.append(variant_id)  # Store the simulated index
        ref = random.choice(["A", "T", "C", "G"])
        alt = random.choice([letter for letter in ["A", "T", "C", "G"] if letter != ref])
        qual = random.randint(0, 100)
        filter_ = "PASS"
        info = "."
        format_ = "GT"
        samples = []
        for _ in range(num_samples):
            genotype = random.choice(["0|0", "0|1", "1|0", "1|1"])
            samples.append(genotype)
        record = [chrom, str(pos), variant_id, ref, alt, str(qual), filter_, info, format_]
        record.extend(samples)
        vcf_data.append("\t".join(record))
    # Convert to pandas DataFrame for further processing
    vcf_str = "\n".join(vcf_data)
    vcf_df = pd.read_csv(StringIO(vcf_str), delimiter="\t")
    return vcf_df, simulated_indices


def write_sim_vcf_with_comments(vcf_data: pd.DataFrame):
    """Add a specific number of comment lines to test header is removed properly."""
    num_comment_lines = 107
    comment_lines = [f"# Comment line {i}" for i in range(1, num_comment_lines + 1)]
    sim_vcf_with_comments = "\n".join(comment_lines) + "\n" + "\n".join(vcf_data)
    # Save the simulated VCF as a TSV file
    with open("simulated.vcf", "w") as file:
        file.write(sim_vcf_with_comments)
