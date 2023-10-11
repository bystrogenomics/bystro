import pytest

from opensearchpy import OpenSearch
from bystro.utils.config import get_opensearch_config
from bystro.search.utils.opensearch import gather_opensearch_args
from bystro.proteomics.annotation_interface import get_samples_and_genes_from_query


@pytest.mark.integration
def test_foo():
    opensearch_config = gather_opensearch_args(get_opensearch_config())
    opensearch_client = OpenSearch(**opensearch_config)
    user_query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    # this index points to a particular annotation file derived from
    # trim_trio_vep.vcf on bystro-dev.  If the index is no longer
    # present, this test may fail for no fault of the tested code's own.
    index_name = "64c889415acb6d3b3e40e07b_63ddc9ce1e740e0020c39928"
    samples_genes_df = get_samples_and_genes_from_query(user_query_string, index_name, opensearch_client)
    assert samples_genes_df.shape == (1191, 3)
    assert {"1805", "1847", "4805"} == set(samples_genes_df.sample_id.unique())
    assert 689 == len(samples_genes_df.gene_name.unique())
    assert {1, 2} == set(samples_genes_df.dosage.unique())
