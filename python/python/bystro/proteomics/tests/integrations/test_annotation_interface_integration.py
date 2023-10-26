import asyncio
import logging
import time

import pytest
from bystro.search.index.handler import go as upload_and_index_vcf
from bystro.search.utils.opensearch import gather_opensearch_args
from bystro.proteomics.annotation_interface import get_annotation_result_from_query
from bystro.utils.config import (
    BYSTRO_PROJECT_ROOT,
    get_mapping_config,
    get_opensearch_config,
    ReferenceGenome,
)
from flaky import flaky
from opensearchpy import OpenSearch

logger = logging.getLogger(__file__)

opensearch_config = get_opensearch_config()
opensearch_client = OpenSearch(**gather_opensearch_args(opensearch_config))


def ensure_annotation_file_present(annotation_filename: str) -> None:
    if not opensearch_client.indices.exists(annotation_filename):
        msg = f"Didn't find {annotation_filename} on server, uploading..."
        logger.debug(msg)
        upload_test_annotation_file(annotation_filename)
    else:
        msg = f"Found {annotation_filename} on server, proceeding..."
        logger.debug(msg)


def upload_test_annotation_file(annotation_filename: str) -> None:
    mapping_config = get_mapping_config(ReferenceGenome.hg38)
    tar_path = (
        BYSTRO_PROJECT_ROOT / "python/python/bystro/proteomics/tests/integrations/trio_trim_vep_vcf.tar"
    )
    assert tar_path.exists()
    asyncio.run(
        upload_and_index_vcf(annotation_filename, mapping_config, opensearch_config, str(tar_path))
    )
    # after uploading, some additional time is required on the annotator's end before the results
    # are available to query.  Pause here for a few seconds in order to avoid prematurely querying
    # the annotation file.  This workaround should become obsolete with the deployment of
    # https://github.com/bystrogenomics/bystro/pull/310, at which point the time.sleep call can be
    # deleted.
    time.sleep(3)


@pytest.mark.integration()
@flaky(max_runs=2, min_passes=1)
def test_get_annotation_result_from_query_integration():
    index_name = "trio_trim_vep_annotation_for_integration_testing_purposes"
    ensure_annotation_file_present(index_name)
    user_query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    samples_genes_df = get_annotation_result_from_query(user_query_string, index_name, opensearch_client)
    assert samples_genes_df.shape == (1610, 7)
    assert {"1805", "1847", "4805"} == set(samples_genes_df.sample_id.unique())
    assert 689 == len(samples_genes_df.gene_name.unique())
    # it's awkward to test for equality of NaN objects, so fill them
    # and compare the filled sets instead.
    MISSING_GENO_VALUE = -1
    expected_dosage_values = {1, 2, MISSING_GENO_VALUE}
    actual_dosage_values = set(samples_genes_df.dosage.fillna(MISSING_GENO_VALUE).unique())
    assert expected_dosage_values == actual_dosage_values
