import copy
from pathlib import Path
import msgspec
import numpy as np
import pandas as pd
import pytest
import mock
from bystro.proteomics.annotation_interface import (
    process_query_response,
    join_annotation_result_to_proteomics_dataset,
    get_annotation_result_from_query,
    async_get_annotation_result_from_query,
    SAMPLE_GENERATED_COLUMN,
    ALWAYS_INCLUDED_FIELDS,
    SAMPLE_COLUMNS,
    LINK_GENERATED_COLUMN,
)
from bystro.proteomics.fragpipe_tandem_mass_tag import (
    ABUNDANCE_COLS,
    TandemMassTagDataset,
)

from bystro.api.auth import CachedAuth

import json

TEST_LEGACY_RESPONSE_PATH = Path(__file__).parent / "test_legacy_response.dat"

with TEST_LEGACY_RESPONSE_PATH.open("rb") as f:
    TEST_LEGACY_RESPONSE = msgspec.msgpack.decode(f.read())  # noqa: S301 (data is safe)

TEST_RESPONSES_WITH_SAMPLES_PATH = Path(__file__).parent / "example_query_response_with_samples.json"
TEST_RESPONSES_WITHOUT_SAMPLES_PATH = (
    Path(__file__).parent / "example_query_response_without_samples.json"
)

with TEST_RESPONSES_WITH_SAMPLES_PATH.open("r") as f:
    TEST_RESPONSES_WITH_SAMPLES = json.load(f)

with TEST_RESPONSES_WITHOUT_SAMPLES_PATH.open("r") as f:
    TEST_RESPONSES_WITHOUT_SAMPLES = json.load(f)


class MockAsyncOpenSearchLegacy:
    def __init__(self):
        self.has_sent_hits = False

    async def search(self, *_args, **_kw_args) -> dict:
        if not self.has_sent_hits:
            self.has_sent_hits = True
            return TEST_LEGACY_RESPONSE

        response = copy.deepcopy(TEST_LEGACY_RESPONSE)
        response["hits"]["total"] = 0
        response["hits"]["hits"] = []

        return response

    async def count(self, *_args, **_kw_args) -> dict:
        return {"count": 1}

    async def create_point_in_time(self, *_args, **_kw_args) -> dict:
        return {"pit_id": "12345"}

    async def delete_point_in_time(self, *_args, **_kw_args) -> None:
        return

    async def close(self) -> None:
        return


class MockAsyncOpenSearch:
    def __init__(self, responses: list[dict]):
        self.pages_seen = 0
        self.responses = responses

    async def search(self, *_args, **_kw_args) -> dict:
        if self.pages_seen < len(self.responses):
            res = self.responses[self.pages_seen]
        else:
            res = {
                "hits": {
                    "hits": [],
                    "total": 0,
                }
            }
        self.pages_seen += 1

        return res

    async def count(self, *_args, **_kw_args) -> dict:
        return {"count": 1}

    async def create_point_in_time(self, *_args, **_kw_args) -> dict:
        return {"pit_id": "12345"}

    async def delete_point_in_time(self, *_args, **_kw_args) -> None:
        return

    async def close(self) -> None:
        return


@pytest.mark.asyncio
async def test_legacy_get_annotation_results_from_query(mocker):
    mocker.patch(
        "bystro.proteomics.annotation_interface.AsyncOpenSearch",
        return_value=MockAsyncOpenSearchLegacy(),
    )
    # inputRef doesn't exist in the legacy datasets, pre Q1-2024
    mocker.patch("bystro.proteomics.annotation_interface.INPUT_REF_FIELD", "ref")
    mocker.patch(
        "bystro.proteomics.annotation_interface.ALWAYS_INCLUDED_FIELDS",
        [
            "chrom",
            "pos",
            "vcfPos",
            "ref",
            "alt",
            "type",
            "id",
        ],
    )

    query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = "mock_index_name"

    samples_and_genes_df = await async_get_annotation_result_from_query(
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
    assert (1405, 52) == samples_and_genes_df.shape


@pytest.mark.asyncio
async def test_get_annotation_results_from_query_with_samples(mocker):
    mocker.patch(
        "bystro.proteomics.annotation_interface.AsyncOpenSearch",
        return_value=MockAsyncOpenSearch(TEST_RESPONSES_WITH_SAMPLES),
    )

    query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = "mock_index_name"

    samples_and_genes_df = await async_get_annotation_result_from_query(
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

    assert (397, 11) == samples_and_genes_df.shape


@pytest.mark.asyncio
async def test_get_annotation_results_from_query_without_samples(mocker):
    mocker.patch(
        "bystro.proteomics.annotation_interface.AsyncOpenSearch",
        return_value=MockAsyncOpenSearch(TEST_RESPONSES_WITHOUT_SAMPLES),
    )

    query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = "mock_index_name"

    samples_and_genes_df = await async_get_annotation_result_from_query(
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
    assert (3698, 9) == samples_and_genes_df.shape

import json
def test_process_response():
    all_hits = []
    for batch in TEST_RESPONSES_WITH_SAMPLES:
        all_hits.extend(batch["hits"]["hits"])
    print(json.dumps(all_hits, indent=2))
    ans = process_query_response(all_hits)

    assert (177, 12) == ans.shape

    # # By default we include all fields in the index
    assert set(ans.columns) == set(
        ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN] + SAMPLE_COLUMNS + ["gnomad.exomes.AF", "refSeq.name2"]
    )

    # # # If we specify fields, only the defaults + those specified should be included
    # # # if the fields specified already exist in the default, they should not be duplicated
    # ans = process_query_response(all_hits, fields=SAMPLE_COLUMNS)

    # assert (177, 11) == ans.shape
    # assert set(ans.columns) == set(ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN] + SAMPLE_COLUMNS)

    # # If we melt, we drop SAMPLE_COLUMNS, in favor of ['sample', 'dosage']
    # melted = process_query_response(all_hits, melt_by_samples=True)
    # assert (397, 10) == melted.shape
    # assert set(melted.columns) == set(
    #     ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN] + ["sample", "dosage"]
    # )

    # assert {"1805", "1847", "4805"} == set(melted[SAMPLE_GENERATED_COLUMN].unique())

    # # For every sample that was a heterozygote, we should have dosage 1
    # for _, row in ans.iterrows():
    #     locus = row[LINK_GENERATED_COLUMN]
    #     melted_rows = melted[melted[LINK_GENERATED_COLUMN] == locus]
    #     heterozygotes = row["heterozygotes"]
    #     homozygotes = row["homozygotes"]
    #     missing = row["missingGenos"]

    #     if heterozygotes is not None:
    #         if not isinstance(heterozygotes, list):
    #             heterozygotes = [heterozygotes]

    #         for sample in heterozygotes:
    #             assert 1 == melted_rows[melted_rows["sample"] == str(sample)]["dosage"].to_numpy()[0]

    #     if homozygotes is not None:
    #         if not isinstance(homozygotes, list):
    #             homozygotes = [homozygotes]

    #         for sample in homozygotes:
    #             assert 2 == melted_rows[melted_rows["sample"] == str(sample)]["dosage"].to_numpy()[0]

    #     if missing is not None:
    #         if not isinstance(missing, list):
    #             missing = [missing]

    #         for sample in missing:
    #             assert -1 == melted_rows[melted_rows["sample"] == str(sample)]["dosage"].to_numpy()[0]


@pytest.mark.asyncio
async def test_join_annotation_result_to_proteomics_dataset(mocker):
    mocker.patch(
        "bystro.proteomics.annotation_interface.AsyncOpenSearch",
        return_value=MockAsyncOpenSearch(TEST_RESPONSES_WITH_SAMPLES),
    )

    query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = "foo"

    query_result_df = await async_get_annotation_result_from_query(
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

    shared_proteomics_sample_ids = sorted(set(query_result_df.sample_id))[:2]
    all_proteomics_sample_ids = shared_proteomics_sample_ids + [f"sample_id{i}" for i in range(10)]

    shared_proteomics_gene_names = sorted(set(query_result_df.gene_name))[:100]
    all_proteomics_gene_names = shared_proteomics_gene_names + [f"gene_name{i}" for i in range(10)]

    final_abundance_cols = [col for col in ABUNDANCE_COLS if col != "Index"]
    columns = final_abundance_cols + all_proteomics_sample_ids
    abundance_df = pd.DataFrame(
        np.random.random((len(all_proteomics_gene_names), len(columns))),
        columns=columns,
        index=all_proteomics_gene_names,
    )
    mock_tmt_dataset = TandemMassTagDataset(abundance_df=abundance_df, annotation_df=pd.DataFrame())

    joined_df = join_annotation_result_to_proteomics_dataset(query_result_df, mock_tmt_dataset)

    assert (478, 19) == joined_df.shape
    assert set(shared_proteomics_sample_ids) == set(joined_df.sample_id)
    assert set(shared_proteomics_gene_names) == set(joined_df.gene_name)
