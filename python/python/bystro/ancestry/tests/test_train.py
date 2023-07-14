"""Tests for ancestry model training code."""

import pandas as pd
import random
import os

from bystro.ancestry.train import (
    load_1kgp_vcf,
    process_vcf_for_pc_transformation
)

from bystro.ancestry import train

## this is not a very useful test, really just testing that train
## loaded successfully, didn't fail on import

def test_module_is_loaded_correctly():
    assert "Ancestry" in train.__doc__

#Pick num of samples for simulated VCF and loadings data
num_samples=10
def generate_random_vcf_index():
    random_number_1_22 = random.randint(1, 22)
    random_number_1_million = random.randint(1, 1000000)
    random_ref = random.choice(['A', 'T', 'C', 'G'])
    random_alt = random.choice([letter for letter in ['A', 'T', 'C', 'G'] if letter != random_ref])

    return f"{random_number_1_22}:{random_number_1_million}:{random_ref}:{random_alt}"

def generate_simulated_vcf(num_samples):
    header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    sample_ids = [f"SampleID{i+1}" for i in range(num_samples)]
    header.extend(sample_ids)
    
    vcf_data = []

    simulated_indices = []
    vcf_data.append('\t'.join(header))
    
    for variant_id in range(1, 101):
        chrom = 'chr1'
        pos = variant_id
        variant_id = generate_random_vcf_index()  # Using simulated index as variant ID
        simulated_indices.append(variant_id)  # Store the simulated index
        ref = random.choice(['A', 'T', 'C', 'G'])
        alt = random.choice([letter for letter in ['A', 'T', 'C', 'G'] if letter != ref])
        qual = random.randint(0, 100)
        filter_ = "PASS"
        info = "."
        format_ = "GT"
        samples = []
        for _ in range(num_samples):
            genotype = random.choice(['0|0', '0|1', '1|0', '1|1'])
            samples.append(genotype)
        record = [chrom, str(pos), variant_id, ref, alt, str(qual), filter_, info, format_]
        record.extend(samples)
        vcf_data.append('\t'.join(record))
    num_comment_lines = 107
    comment_lines = ['# Comment line {}'.format(i) for i in range(1, num_comment_lines + 1)]
    sim_vcf_with_comments = '\n'.join(comment_lines) + '\n' + '\n'.join(vcf_data)
    # Save the simulated VCF as a TSV file
    with open('simulated.vcf', 'w') as file:
        file.write(sim_vcf_with_comments)
    return sim_vcf_with_comments, simulated_indices

def test_load_and_process_vcf():
    """Tests gnomad-filtered 1kgp vcf loading and processing"""
    # Generate simulated VCF and loadings data
    simulated_vcf, simulated_indices = generate_simulated_vcf(num_samples)
    #sim_loadings = pd.DataFrame(data=random.random(), index=simulated_indices, columns=[f"PC{i+1}" for i in range(30)])
    sim_filepath = 'simulated.vcf'
    
    # Call load_1kgp_vcf function with the simulated VCF file path
    loaded_result = load_1kgp_vcf(sim_filepath)
    
    # Assert that the loaded_result is a DataFrame
    assert isinstance(loaded_result, pd.DataFrame)

    # Assert that the loaded_result has expected columns after processing
    expected_columns = ['Chromosome', 'Position']
    for column in expected_columns:
        assert column in loaded_result.columns

    # Call process_vcf_for_pc_transformation function with the loaded_result DataFrame
    processed_result = process_vcf_for_pc_transformation(loaded_result)

    # Assert that the processed_result DataFrame only contains values of 0, 1, or 2
    valid_values = [0, 1, 2]
    for column in processed_result.columns:
        for value in processed_result[column]:
            assert value in valid_values
            
    # Delete the simulated VCF file
    os.remove(sim_filepath)
