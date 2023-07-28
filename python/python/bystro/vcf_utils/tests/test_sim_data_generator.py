"""Test simulated data."""

import pandas as pd
from bystro.vcf_utils.sim_data_generator import (
    generate_random_vcf_index, 
    generate_simulated_vcf
)

HEADER_COLS = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]


def test_generate_random_vcf_index():
    """Test expected index components have correct parameters."""
    random_vcf_index = generate_random_vcf_index()
    assert isinstance(random_vcf_index, str)
    parts = random_vcf_index.split(":")
    assert len(parts) == 4
    assert parts[0].isdigit()
    assert 1 <= int(parts[0]) <= 22
    assert parts[1].isdigit()
    assert 1 <= int(parts[1]) <= 1000000
    assert parts[2] in ["A", "T", "C", "G"]
    assert parts[3] in ["A", "T", "C", "G"]
    assert parts[2] != parts[3]


def test_generate_simulated_vcf():
    """Test generate_simulated_vcf."""
    num_samples = 10
    num_vars = 10
    simulated_vcf, simulated_indices = generate_simulated_vcf(num_samples, num_vars)
    assert isinstance(simulated_vcf, pd.DataFrame)
    assert isinstance(simulated_indices, list)
    assert len(simulated_indices) == num_vars
    assert len(simulated_vcf.columns) == num_samples + len(HEADER_COLS)
