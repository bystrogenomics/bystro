from pathlib import Path

import msgspec
import numpy as np
import pandas as pd

from bystro.proteomics.annotation_interface import (
    _process_response,
    join_annotation_result_to_proteomics_dataset,
    get_annotation_result_from_query,
)

from bystro.proteomics.fragpipe_tandem_mass_tag import (
    ABUNDANCE_COLS,
    TandemMassTagDataset,
)

TEST_RESPONSE_FILENAME = Path(__file__).parent / "test_response.dat"

with TEST_RESPONSE_FILENAME.open("rb") as f:
    TEST_RESPONSE = msgspec.msgpack.decode(f.read())  # noqa: S301 (data is safe)


class MockAsyncOpenSearch:
    def __init__(self):
        self.has_sent_hits = False

    async def search(self, *_args, **_kw_args) -> dict:
        if not self.has_sent_hits:
            self.has_sent_hits = True
            return TEST_RESPONSE

        response = TEST_RESPONSE.copy()
        response["hits"] = {"hits": []}

        return response

    async def count(self, *_args, **_kw_args) -> dict:
        return {"count": 1}

    async def create_point_in_time(self, *_args, **_kw_args) -> dict:
        return {"pit_id": "12345"}

    async def delete_point_in_time(self, *_args, **_kw_args) -> None:
        return

    async def close(self) -> None:
        return


def test_get_annotation_results_from_query(mocker):
    mocker.patch(
        "bystro.proteomics.annotation_interface.AsyncOpenSearch",
        return_value=MockAsyncOpenSearch(),
    )
    query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = "mock_index_name"

    samples_and_genes_df = get_annotation_result_from_query(
        query_string,
        index_name,
        cluster_opensearch_config={
            "connection": {
                "nodes": ["http://localhost:9200"],
                "request_timeout": 1200,
                "use_ssl": False,
                "verify_certs": False,
            },
        },
    )
    assert (4645, 18) == samples_and_genes_df.shape


def test_process_response():
    ans = _process_response(TEST_RESPONSE["hits"]["hits"])

    assert (4645, 18) == ans.shape
    assert {"1805", "1847", "4805"} == set(ans.sample_id.unique())
    assert 689 == len(ans.gene_name.unique())
    # it's awkward to test for equality of NaN objects, so fill them
    # and compare the filled sets instead.
    MISSING_GENO_VALUE = -1
    expected_dosage_values = {1.0, 2.0, MISSING_GENO_VALUE}
    actual_dosage_values = set(ans.dosage.fillna(MISSING_GENO_VALUE).unique())
    assert expected_dosage_values == actual_dosage_values


def test_join_annotation_result_to_proteomics_dataset(mocker):
    mocker.patch(
        "bystro.proteomics.annotation_interface.AsyncOpenSearch",
        return_value=MockAsyncOpenSearch(),
    )

    # Step 1: Get an annotation query result
    query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = "foo"

    query_result_df = get_annotation_result_from_query(
        query_string,
        index_name,
        cluster_opensearch_config={
            "connection": {
                "nodes": ["http://localhost:9200"],
                "request_timeout": 1200,
                "use_ssl": False,
                "verify_certs": False,
            },
        },
    )

    # Step 2: Construct a proteomics dataset with some shared sampled_ids and gene_names
    shared_proteomics_sample_ids = sorted(set(query_result_df.sample_id))[:2]
    all_proteomics_sample_ids = shared_proteomics_sample_ids + [f"sample_id{i}" for i in range(10)]

    shared_proteomics_gene_names = sorted(set(query_result_df.gene_name))[:100]
    all_proteomics_gene_names = shared_proteomics_gene_names + [f"gene_name{i}" for i in range(10)]

    # we're constructing the abundance df directly as the
    # TandemMassTagDataset expects it, and so need to drop the 'Index'
    # column
    final_abundance_cols = [col for col in ABUNDANCE_COLS if col != "Index"]
    columns = final_abundance_cols + all_proteomics_sample_ids
    abundance_df = pd.DataFrame(
        np.random.random((len(all_proteomics_gene_names), len(columns))),
        columns=columns,
        index=all_proteomics_gene_names,
    )
    mock_tmt_dataset = TandemMassTagDataset(abundance_df=abundance_df, annotation_df=pd.DataFrame())

    # Step 3: join the two togoether
    joined_df = join_annotation_result_to_proteomics_dataset(query_result_df, mock_tmt_dataset)

    # Step 4: Test the joined result
    assert (478, 19) == joined_df.shape
    assert set(shared_proteomics_sample_ids) == set(joined_df.sample_id)
    assert set(shared_proteomics_gene_names) == set(joined_df.gene_name)
