"""Test simulated vcf data."""

import pandas as pd
from bystro.vcf_utils.simulate_random_vcf import (
    generate_random_vcf_index,
    generate_simulated_vcf,
    convert_sim_vcf_to_df,
    HEADER_COLS,
    add_comment_lines_to_sim_vcf,
)

pd.options.future.infer_string = True  # type: ignore


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
    simulated_vcf = generate_simulated_vcf(num_samples, num_vars)
    sim_vcf_df = convert_sim_vcf_to_df(simulated_vcf)
    assert isinstance(sim_vcf_df, pd.DataFrame)
    assert len(sim_vcf_df.index) == num_vars
    assert len(sim_vcf_df.columns) == num_samples + len(HEADER_COLS)


def test_add_comment_lines_to_sim_vcf():
    """Test comment lines are added correctly"""
    num_samples = 10
    num_vars = 10
    simulated_vcf = generate_simulated_vcf(num_samples, num_vars)
    vcf_with_comments = add_comment_lines_to_sim_vcf(simulated_vcf)
    num_comment_lines = 107
    assert vcf_with_comments.count("# Comment line") == num_comment_lines


def test_convert_sim_vcf_to_df():
    """Test simulated vcf df is same as original vcf"""
    num_samples = 10
    num_vars = 10
    simulated_vcf = generate_simulated_vcf(num_samples, num_vars)
    vcf_df = convert_sim_vcf_to_df(simulated_vcf)
    vcf_str = vcf_df.to_csv(index=False, sep="\t", lineterminator="").strip()
    assert vcf_str == simulated_vcf
    assert vcf_df.shape == (num_vars, num_samples + len(HEADER_COLS))
