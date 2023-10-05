import pytest

from bystro.proteomics.annotation_interface import (
    get_samples_and_genes,
)


@pytest.mark.network()
def test_get_samples_and_genes():
    """Test annotation interface against bystro-dev."""
    user_query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    SAMPLE_INDEX_NAME = "64c889415acb6d3b3e40e07b_63ddc9ce1e740e0020c39928"
    samples_and_genes_df = get_samples_and_genes(user_query_string, SAMPLE_INDEX_NAME)
    assert 1191 == len(samples_and_genes_df)
