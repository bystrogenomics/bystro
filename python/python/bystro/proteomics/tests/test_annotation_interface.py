import copy
from pathlib import Path
import json

import msgspec
import pytest

from bystro.proteomics.annotation_interface import (
    DOSAGE_GENERATED_COLUMN,
    process_query_response,
    join_annotation_result_to_proteomic_dataset,
    async_get_annotation_result_from_query,
    SAMPLE_GENERATED_COLUMN,
    ALWAYS_INCLUDED_FIELDS,
    SAMPLE_COLUMNS,
    LINK_GENERATED_COLUMN,
)

from bystro.proteomics.fragpipe_tandem_mass_tag import (
    load_tandem_mass_tag_dataset,
    FRAGPIPE_RENAMED_COLUMNS,
    FRAGPIPE_SAMPLE_COLUMN,
    FRAGPIPE_GENE_GENE_NAME_COLUMN_RENAMED,
    FRAGPIPE_SAMPLE_INTENSITY_COLUMN,
)

from bystro.proteomics.somascan import SomascanDataset, ADAT_SAMPLE_ID_COLUMN

TEST_LEGACY_RESPONSE_PATH = Path(__file__).parent / "test_legacy_response.dat"

with TEST_LEGACY_RESPONSE_PATH.open("rb") as f:
    TEST_LEGACY_RESPONSE = msgspec.msgpack.decode(f.read())  # noqa: S301 (data is safe)

TEST_RESPONSES_WITH_SAMPLES_PATH = Path(__file__).parent / "example_query_response_with_samples.json"
TEST_RESPONSES_WITHOUT_SAMPLES_PATH = (
    Path(__file__).parent / "example_query_response_without_samples.json"
)

with TEST_RESPONSES_WITH_SAMPLES_PATH.open("r") as f:  # type: ignore
    TEST_RESPONSES_WITH_SAMPLES = json.load(f)

with TEST_RESPONSES_WITHOUT_SAMPLES_PATH.open("r") as f:  # type: ignore
    TEST_RESPONSES_WITHOUT_SAMPLES = json.load(f)


class MockAsyncOpenSearchLegacy:
    def __init__(self):
        self.has_sent_hits = False

    async def search(self, *_args, **_kw_args) -> dict:
        if not self.has_sent_hits:
            self.has_sent_hits = True
            return copy.deepcopy(TEST_LEGACY_RESPONSE)

        return {
            "hits": {
                "hits": [],
                "total": 0,
            }
        }

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
        self.responses = copy.deepcopy(responses)

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

        return copy.deepcopy(res)

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

    # We melt by default
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

    assert (397, 12) == samples_and_genes_df.shape

    assert list(samples_and_genes_df.columns) == ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN] + [
        SAMPLE_GENERATED_COLUMN,
        DOSAGE_GENERATED_COLUMN,
    ] + ["gnomad.exomes.AF", "refSeq.name2"]


@pytest.mark.asyncio
async def test_get_annotation_results_from_query_with_sample_no_melt(mocker):
    mocker.patch(
        "bystro.proteomics.annotation_interface.AsyncOpenSearch",
        return_value=MockAsyncOpenSearch(TEST_RESPONSES_WITH_SAMPLES),
    )

    query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    index_name = "mock_index_name"

    # We melt by default
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
        melt_samples=False,
    )

    assert (177, 13) == samples_and_genes_df.shape

    additional_columns = SAMPLE_COLUMNS + ["gnomad.exomes.AF", "refSeq.name2"]
    additional_columns.sort()
    assert (
        list(samples_and_genes_df.columns)
        == ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN] + additional_columns
    )


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

    assert (3698, 10) == samples_and_genes_df.shape
    assert list(samples_and_genes_df.columns) == ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN] + [
        "gnomad.exomes.AF",
        "refSeq.name2",
    ]


def test_process_response():
    all_hits = []
    for batch in TEST_RESPONSES_WITH_SAMPLES:
        all_hits.extend(batch["hits"]["hits"])

    ans = process_query_response(copy.deepcopy(all_hits))

    assert (177, 13) == ans.shape

    default_fields = ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN]
    additional_fields = SAMPLE_COLUMNS + [
        "gnomad.exomes.AF",
        "refSeq.name2",
    ]
    additional_fields.sort()
    # # By default we include all fields in the index
    assert list(ans.columns) == default_fields + additional_fields

    # If we specify fields, only the defaults + those specified should be included
    # if the fields specified already exist in the default, they should not be duplicated
    # We don't always provide sample columns, but do always provide a generated locus/link column
    ans = process_query_response(copy.deepcopy(all_hits), fields=ALWAYS_INCLUDED_FIELDS)

    assert (177, 8) == ans.shape
    assert list(ans.columns) == ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN]

    # but if the fields don't exist in the default set, we will add them
    ans = process_query_response(copy.deepcopy(all_hits), fields=["refSeq.name2"])

    assert (177, 9) == ans.shape
    assert list(ans.columns) == ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN] + ["refSeq.name2"]

    # If we melt, we drop SAMPLE_COLUMNS, in favor of ['sample', 'dosage']
    melted = process_query_response(copy.deepcopy(all_hits), melt_samples=True)
    assert (397, 12) == melted.shape
    assert list(melted.columns) == ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN] + [
        SAMPLE_GENERATED_COLUMN,
        DOSAGE_GENERATED_COLUMN,
    ] + ["gnomad.exomes.AF", "refSeq.name2"]

    assert {"1805", "1847", "4805"} == set(melted[SAMPLE_GENERATED_COLUMN].unique())

    ans = process_query_response(copy.deepcopy(all_hits), melt_samples=False, fields=SAMPLE_COLUMNS)
    # # For every sample that was a heterozygote, we should have dosage 1
    for _, row in ans.iterrows():
        locus = row[LINK_GENERATED_COLUMN]
        melted_rows = melted[melted[LINK_GENERATED_COLUMN] == locus]
        heterozygotes = row["heterozygotes"]
        homozygotes = row["homozygotes"]
        missing = row["missingGenos"]

        if heterozygotes is not None:
            if not isinstance(heterozygotes, list):
                heterozygotes = [heterozygotes]

            for sample in heterozygotes:
                assert (
                    1
                    == melted_rows[melted_rows[SAMPLE_GENERATED_COLUMN] == str(sample)][
                        DOSAGE_GENERATED_COLUMN
                    ].to_numpy()[0]
                )

        if homozygotes is not None:
            if not isinstance(homozygotes, list):
                homozygotes = [homozygotes]

            for sample in homozygotes:
                assert (
                    2
                    == melted_rows[melted_rows[SAMPLE_GENERATED_COLUMN] == str(sample)][
                        DOSAGE_GENERATED_COLUMN
                    ].to_numpy()[0]
                )

        if missing is not None:
            if not isinstance(missing, list):
                missing = [missing]

            for sample in missing:
                assert (
                    -1
                    == melted_rows[melted_rows[SAMPLE_GENERATED_COLUMN] == str(sample)][
                        DOSAGE_GENERATED_COLUMN
                    ].to_numpy()[0]
                )


@pytest.mark.asyncio
async def test_join_annotation_result_to_fragpipe_dataset(mocker):
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
        explode_field="refSeq.name2",
    )

    assert (582, 12) == query_result_df.shape

    sample_ids = query_result_df[SAMPLE_GENERATED_COLUMN].unique()

    abundance_file = str(Path(__file__).parent / "example_abundance_gene_MD.tsv")
    experiment_file = str(Path(__file__).parent / "example_experiment_annotation_file.tsv")
    tmt_dataset = load_tandem_mass_tag_dataset(abundance_file, experiment_file)

    sample_names = list(tmt_dataset.annotation_df.index)[0 : sample_ids.shape[0]]

    # replace the sample ids with the sample names
    replacements = {sample_id: sample_name for sample_id, sample_name in zip(sample_ids, sample_names)}
    query_result_df[SAMPLE_GENERATED_COLUMN] = query_result_df[SAMPLE_GENERATED_COLUMN].replace(
        replacements
    )

    joined_df = join_annotation_result_to_proteomic_dataset(
        query_result_df, tmt_dataset, proteomic_join_column=FRAGPIPE_GENE_GENE_NAME_COLUMN_RENAMED
    )

    assert (90, 17) == joined_df.shape

    retained_fragpipe_columns = []
    for name in FRAGPIPE_RENAMED_COLUMNS:
        if name in [FRAGPIPE_SAMPLE_COLUMN, FRAGPIPE_GENE_GENE_NAME_COLUMN_RENAMED]:
            continue
        retained_fragpipe_columns.append(name)

    retained_fragpipe_columns.append(FRAGPIPE_SAMPLE_INTENSITY_COLUMN)
    assert list(joined_df.columns) == list(query_result_df.columns) + retained_fragpipe_columns


@pytest.mark.asyncio
async def test_join_annotation_result_to_somascan_dataset(mocker):
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
        explode_field="refSeq.name2",
    )

    assert (582, 12) == query_result_df.shape

    sample_ids = query_result_df[SAMPLE_GENERATED_COLUMN].unique()

    adat_file = str(Path(__file__).parent / "example_data_v4.1_plasma.adat")
    somascan_dataset = SomascanDataset.from_paths(adat_file)

    sample_names = list(somascan_dataset.adat.index.to_frame()[ADAT_SAMPLE_ID_COLUMN].values)[
        0 : sample_ids.shape[0]
    ]
    replacements = {sample_id: sample_name for sample_id, sample_name in zip(sample_ids, sample_names)}
    query_result_df[SAMPLE_GENERATED_COLUMN] = query_result_df[SAMPLE_GENERATED_COLUMN].replace(
        replacements
    )

    joined_df_soma = join_annotation_result_to_proteomic_dataset(query_result_df, somascan_dataset)

    assert (131, 71) == joined_df_soma.shape
