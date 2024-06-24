"""VCF index and data simulator for testing analysis modules."""

import random
from io import StringIO

import pandas as pd

pd.options.future.infer_string = True  # type: ignore

HEADER_COLS = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]


def generate_random_vcf_index() -> tuple[int, int, str, str]:
    """Generate autosomal chr vcf variant IDs to use for testing."""
    random_chr = random.randint(1, 22)
    random_pos = random.randint(1, 1000000)
    random_ref = random.choice(["A", "T", "C", "G"])
    random_alt = random.choice([letter for letter in ["A", "T", "C", "G"] if letter != random_ref])
    return random_chr, random_pos, random_ref, random_alt


def generate_simulated_vcf(num_samples: int, num_vars: int) -> str:
    """VCF simulator for testing analysis modules."""
    # This is a first pass - could include more sophisticated choices for possible values in future
    sample_ids = [f"SampleID{i+1}" for i in range(num_samples)]
    header = HEADER_COLS + sample_ids
    vcf_data = []
    simulated_indices = []
    vcf_data.append("#" + "\t".join(header))
    for variant in range(num_vars):
        # Use index simulator to generate random chrom,pos,ref,alt
        random_chr, random_pos, random_ref, random_alt = generate_random_vcf_index()
        variant_id = f"{random_chr}:{random_pos}:{random_ref}:{random_alt}"
        simulated_indices.append(variant_id)
        qual = random.randint(0, 100)
        filter_ = "PASS"
        info = "."
        format_ = "GT"
        samples = []
        record = [
            str(random_chr),
            str(random_pos),
            variant_id,
            random_ref,
            random_alt,
            str(qual),
            filter_,
            info,
            format_,
        ]
        for _ in range(num_samples):
            genotype = random.choice(["0|0", "0|1", "1|0", "1|1"])
            samples.append(genotype)
        record.extend(samples)
        vcf_data.append("\t".join(record))
        vcf_str = "\n".join(vcf_data)
    return vcf_str


def convert_sim_vcf_to_df(vcf_str: str) -> pd.DataFrame:
    """Convert to vcf to pandas DataFrame for further processing"""
    vcf_df = pd.read_csv(StringIO(vcf_str), delimiter="\t")
    return vcf_df


def add_comment_lines_to_sim_vcf(vcf_str: str) -> str:
    """Add a specific number of comment lines to test header is removed properly."""
    num_comment_lines = 107
    comment_lines = [f"# Comment line {i}" for i in range(1, num_comment_lines + 1)]
    sim_vcf_with_comments = "\n".join(comment_lines) + "\n" + "\n".join(vcf_str)
    return sim_vcf_with_comments


def write_out_sim_vcf(sim_vcf_name: str, sim_vcf_with_comments: str):
    """Save the simulated VCF as a TSV file"""
    with open(sim_vcf_name, "w") as file:
        file.write(sim_vcf_with_comments)
