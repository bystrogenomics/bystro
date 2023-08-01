"""Test simulated vcf data."""

import pandas as pd
from bystro.vcf_utils.sim_data_generator import (
    generate_random_vcf_index,
    generate_simulated_vcf,
    convert_sim_vcf_to_pd,
    HEADER_COLS,
)


def test_generate_random_vcf_index():
    """Test expected index components have correct parameters."""
    random_chr, random_pos, random_ref, random_alt = generate_random_vcf_index()
    assert 1 <= int(random_chr) <= 22
    assert 1 <= int(random_pos) <= 1000000
    assert random_ref in ["A", "T", "C", "G"]
    assert random_alt in ["A", "T", "C", "G"]
    assert random_ref != random_alt


def test_generate_simulated_vcf():
    """Test generate_simulated_vcf."""
    num_samples = 10
    num_vars = 10
    simulated_vcf, simulated_indices = generate_simulated_vcf(num_samples, num_vars)
    sim_vcf_pd = convert_sim_vcf_to_pd(simulated_vcf)
    assert isinstance(sim_vcf_pd, pd.DataFrame)
    assert isinstance(simulated_indices, list)
    assert len(simulated_indices) == num_vars
    assert len(sim_vcf_pd.columns) == num_samples + len(HEADER_COLS)