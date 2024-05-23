import logging
import time

import pytest
from bystro.search.index.listener import run_handler_with_config
from bystro.search.utils.opensearch import gather_opensearch_args
from bystro.proteomics.annotation_interface import get_annotation_result_from_query
from bystro.utils.config import (
    BYSTRO_PROJECT_ROOT,
    BYSTRO_CONFIG_DIR,
    OPENSEARCH_CONFIG_PATH,
    get_opensearch_config,
    ReferenceGenome,
)
from flaky import flaky
from opensearchpy import OpenSearch

logger = logging.getLogger(__file__)


def ensure_annotation_file_present(index_name: str, opensearch_client: OpenSearch) -> None:
    if not opensearch_client.indices.exists(index_name):
        msg = f"Didn't find {index_name} on server, uploading..."
        logger.debug(msg)
        index_test_annotation_file(index_name)
    else:
        msg = f"Found {index_name} on server, proceeding..."
        logger.debug(msg)


def index_test_annotation_file(index_name: str) -> None:
    mapping_config = str(BYSTRO_CONFIG_DIR / f"{ReferenceGenome.hg19}.mapping.yml")
    annotation_path = str(
        BYSTRO_PROJECT_ROOT
        / "python/python/bystro/proteomics/tests/integrations/trio_trim_vep_vcf.annotation.tsv.gz"
    )

    run_handler_with_config(
        index_name=index_name,
        mapping_config=mapping_config,
        opensearch_config=str(OPENSEARCH_CONFIG_PATH),
        annotation_path=annotation_path,
        no_queue=True,
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
    opensearch_config = get_opensearch_config()
    opensearch_client = OpenSearch(**gather_opensearch_args(opensearch_config))

    index_name = "trio_trim_vep_annotation_for_integration_testing_purposes"
    ensure_annotation_file_present(index_name, opensearch_client)
    query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    samples_genes_df = get_annotation_result_from_query(
        query_string, index_name, cluster_opensearch_config=opensearch_config
    )
    assert samples_genes_df.shape == (1610, 7)
    assert {"1805", "1847", "4805"} == set(samples_genes_df.sample_id.unique())
    assert 689 == len(samples_genes_df.gene_name.unique())
    # it's awkward to test for equality of NaN objects, so fill them
    # and compare the filled sets instead.

    expected_dosage_values = {1, 2, -1}
    actual_dosage_values = set(samples_genes_df.dosage.fillna(-1).unique())
    assert expected_dosage_values == actual_dosage_values
