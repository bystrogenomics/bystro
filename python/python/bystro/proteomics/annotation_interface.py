"""Query an annotation file and return a list of sample_ids and genes meeting the query criteria."""

import copy
import logging
import math
from typing import Any, Callable

import asyncio
from msgspec import Struct
import numpy as np

import pandas as pd
from opensearchpy import AsyncOpenSearch

from bystro.proteomics.fragpipe_tandem_mass_tag import TandemMassTagDataset

from bystro.search.utils.opensearch import gather_opensearch_args

logger = logging.getLogger(__file__)

HETEROZYGOTE_DOSAGE = 1
HOMOZYGOTE_DOSAGE = 2
MISSING_GENO_DOSAGE = np.nan
ONE_DAY = "1d"  # default keep_alive time for opensearch point in time index

# The fields to return for each variant matched by the query
OUTPUT_FIELDS = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "refSeq.name2",
    "homozygotes",
    "heterozygotes",
    "missingGenos",
]


class OpenSearchQueryConfig(Struct):
    """Represent parameters for configuring OpenSearch queries."""

    max_query_size: int = 10_000
    max_slices: int = 1024
    keep_alive: str = ONE_DAY


OPENSEARCH_QUERY_CONFIG = OpenSearchQueryConfig()


def _flatten(xs: Any) -> list[Any]:  # noqa: ANN401 (`Any` is really correct here)
    """Flatten an arbitrarily nested list."""
    if not isinstance(xs, list):
        return [xs]
    return sum([_flatten(x) for x in xs], [])


def _get_nested_field(data, field_path):
    """Recursively fetch nested field values using dot notation."""
    keys = field_path.split(".")
    value = data
    for key in keys:
        try:
            value = value[key]
        except (KeyError, TypeError):
            return None  # Returns None if the field path is not found
    return value


def _get_samples_genes_dosages_from_hit(
    hit: dict[str, Any], additional_fields: list[str] | None = None
) -> pd.DataFrame:
    """Given a document hit, return a dataframe of samples,
    genes and dosages with specified additional fields.
    """

    source = hit["_source"]
    # Base required fields
    chrom = _flatten(source["chrom"])[0]

    pos = _flatten(source["pos"])[0]
    _id = _flatten(source["id"])[0]
    vcf_pos = _flatten(source["vcfPos"])[0]
    input_ref = _flatten(source.get("inputRef", [None]))[0]
    ref = _flatten(source["ref"])[0]
    alt = _flatten(source["alt"])[0]
    gene_names = _flatten(
        source["refSeq"]["name2"]
    )  # guaranteed to be unique or to belong to different transcripts
    transcript_names = _flatten(source["refSeq"]["name"])
    protein_names = _flatten(source["refSeq"].get("protAcc", [None]))
    is_canonical = [
        x == "true" or x is True if x else False for x in _flatten(source["refSeq"].get("isCanonical"))
    ]
    gnomad_genomes_af = _flatten(_get_nested_field(source, "gnomad.genomes.AF"))[0]
    gnomad_exomes_af = _flatten(_get_nested_field(source, "gnomad.exomes.AF"))[0]

    if len(is_canonical) != len(transcript_names):
        is_canonical = [is_canonical[0]] * len(transcript_names)

    if len(protein_names) != len(transcript_names):
        protein_names = [protein_names[0]] * len(transcript_names)

    if len(gene_names) != len(transcript_names):
        gene_names = [gene_names[0]] * len(transcript_names)

    heterozygotes = _flatten(source.get("heterozygotes", []))
    homozygotes = _flatten(source.get("homozygotes", []))
    missing_genos = _flatten(source.get("missingGenos", []))

    heterozygosity = _flatten(source["heterozygosity"])[0]
    homozygosity = _flatten(source["homozygosity"])[0]
    missingness = _flatten(source["missingness"])[0]

    fields_to_add = list(
        filter(
            lambda x: x
            not in [
                "chrom",
                "pos",
                "id",
                "vcfPos",
                "inputRef",
                "ref",
                "alt",
                "refSeq.name2",
                "refSeq.name",
                "refSeq.protAcc",
                "refSeq.isCanonical",
                "heterozygotes",
                "homozygotes",
                "missingGenos",
                "heterozygosity",
                "homozygosity",
                "missingness",
                "gnomad.genomes.AF",
                "gnomad.exomes.AF",
            ],
            additional_fields if additional_fields is not None else [],
        )
    )

    rows = []
    for gene_idx, gene_name in enumerate(gene_names):
        for sample_list, dosage_label in [
            (heterozygotes, 1),
            (homozygotes, 2),
            (missing_genos, -1),
        ]:
            for sample_id in sample_list:
                row = {
                    "sample_id": sample_id,
                    "chrom": chrom,
                    "vcf_pos": vcf_pos,
                    "pos": pos,
                    "id": _id,
                    "ref": ref,
                    "input_ref": input_ref,
                    "alt": alt,
                    "gene_name": gene_name,
                    "transcript_name": transcript_names[gene_idx],
                    "protein_name": protein_names[gene_idx],
                    "is_canonical": is_canonical[gene_idx],
                    "dosage": dosage_label,
                    "heterozygosity": heterozygosity,
                    "homozygosity": homozygosity,
                    "missingness": missingness,
                    "gnomad_genomes_af": gnomad_genomes_af,
                    "gnomad_exomes_af": gnomad_exomes_af,
                }
                # Add additional fields
                for field in fields_to_add:
                    row[field] = _get_nested_field(source, field)

                    if row[field] is not None:
                        row[field] = _flatten(row[field])

                        if len(row[field]) == 1:
                            row[field] = row[field][0]
                        else:
                            row[field] = tuple(row[field])
                rows.append(row)

    return pd.DataFrame(rows)


def _prepare_query_body(query, slice_id, num_slices):
    """Prepare the query body for the slice"""
    body = query.copy()
    body["slice"] = {"id": slice_id, "max": num_slices}
    return body


async def _execute_query(
    client: AsyncOpenSearch, query: dict, additional_fields: list[str] | None = None
) -> pd.DataFrame:
    results: list[dict] = []
    search_after = None  # Initialize search_after for pagination

    # Ensure there is a sort parameter in the query
    if "sort" not in query.get("body", {}):
        query.setdefault("body", {}).update(
            {"sort": [{"_id": "asc"}]}  # Sorting by ID in ascending order
        )

    while True:
        if search_after:
            query["body"]["search_after"] = search_after

        resp = await client.search(**query)

        if not resp["hits"]["hits"]:
            break  # Exit the loop if no more documents are found

        results.extend(resp["hits"]["hits"])

        # Update search_after to the sort value of the last document retrieved
        search_after = resp["hits"]["hits"][-1]["sort"]

    return _process_response(results, additional_fields)


def _process_response(
    hits: list[dict[str, Any]], additional_fields: list[str] | None = None
) -> pd.DataFrame:
    """Postprocess query response from opensearch client."""
    num_hits = len(hits)

    if num_hits == 0:
        return pd.DataFrame()

    samples_genes_dosages_df = pd.concat(
        [_get_samples_genes_dosages_from_hit(hit, additional_fields) for hit in hits]
    )
    # we may have multiple variants per gene in the results, so we
    # need to drop duplicates here.
    return samples_genes_dosages_df.drop_duplicates()


async def _get_num_slices(
    client: AsyncOpenSearch,
    index_name: str,
    query: dict[str, Any],
) -> tuple[int, int]:
    """Count number of hits for the index."""
    get_num_slices_query = query["body"].copy()
    get_num_slices_query.pop("sort", None)
    get_num_slices_query.pop("track_total_hits", None)

    response = await client.count(body=get_num_slices_query, index=index_name)

    n_docs: int = response["count"]
    if n_docs < 1:
        err_msg = (
            f"Expected at least one document in `response['count']`, got response: {response} instead."
        )
        raise RuntimeError(err_msg)

    num_slices_necessary = math.ceil(n_docs / OPENSEARCH_QUERY_CONFIG.max_query_size)
    num_slices_planned = min(num_slices_necessary, OPENSEARCH_QUERY_CONFIG.max_slices)
    return max(num_slices_planned, 1), n_docs


async def _run_annotation_query(
    query: dict[str, Any],
    index_name: str,
    opensearch_config: dict[str, Any],
    additional_fields: list[str] | None = None,
) -> pd.DataFrame:
    """Given query and index contained in SaveJobData, run query and return results as dataframe."""
    search_client_args = gather_opensearch_args(opensearch_config)
    client = AsyncOpenSearch(**search_client_args)

    num_slices, _ = await _get_num_slices(client, index_name, query)

    point_in_time = await client.create_point_in_time(  # type: ignore[attr-defined]
        index=index_name, params={"keep_alive": OPENSEARCH_QUERY_CONFIG.keep_alive}
    )
    try:  # make sure we clean up the PIT index properly no matter what happens in this block
        pit_id = point_in_time["pit_id"]
        query["body"]["pit"] = {"id": pit_id}
        query["body"]["size"] = OPENSEARCH_QUERY_CONFIG.max_query_size
        query_results = []
        for slice_id in range(num_slices):
            slice_query = copy.deepcopy(query)
            if num_slices > 1:
                # Slice queries require max > 1
                slice_query["body"]["slice"] = {"id": slice_id, "max": num_slices}

            query_result = _execute_query(client, query=slice_query, additional_fields=additional_fields)
            query_results.append(query_result)

        res = await asyncio.gather(*query_results)
        return pd.concat(res)
    except Exception as e:
        err_msg = (
            f"Encountered exception: {e!r} while running opensearch_query, "
            "deleting PIT index and exiting.\n"
            f"query: {query}\n"
            f"client: {client}\n"
            f"opensearch_query_config: {OPENSEARCH_QUERY_CONFIG}\n"
        )
        logger.exception(err_msg, exc_info=e)
        raise RuntimeError(err_msg) from e
    finally:
        await client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore[attr-defined]
        await client.close()


async def get_annotation_result_from_query_async(
    user_query_string: str,
    index_name: str,
    opensearch_config: dict[str, Any],
    additional_fields: list[str] | None = None,
) -> pd.DataFrame:
    """Given a query and index, return a dataframe of variant / sample_id records matching query."""
    query = _build_opensearch_query_from_query_string(user_query_string)
    return await _run_annotation_query(query, index_name, opensearch_config, additional_fields)


def get_annotation_result_from_query(
    user_query_string: str,
    index_name: str,
    opensearch_config: dict[str, Any],
    additional_fields: list[str] = [],
) -> pd.DataFrame:
    """Given a query and index, return a dataframe of variant / sample_id records matching query."""
    loop = asyncio.get_event_loop()
    coroutine = get_annotation_result_from_query_async(
        user_query_string, index_name, opensearch_config, additional_fields
    )
    if loop.is_running():
        # If the event loop is already running, use a workaround
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()
    return loop.run_until_complete(coroutine)


def _build_opensearch_query_from_query_string(
    query_string: str, output_fields: list[str] | None = None
) -> dict[str, Any]:
    base_query: dict[str, Any] = {
        "body": {
            "query": {
                "bool": {
                    "filter": {
                        "query_string": {
                            "default_operator": "AND",
                            "query": query_string,
                            "lenient": True,
                            "phrase_slop": 5,
                            "tie_breaker": 0.3,
                        },
                    },
                },
            },
            "sort": "_doc",
        }
    }

    if output_fields is not None:
        base_query["_source_includes"] = output_fields

    return base_query


async def join_annotation_result_to_proteomics_dataset(
    query_result_df: pd.DataFrame,
    tmt_dataset: TandemMassTagDataset,
    get_tracking_id_from_genomic_sample_id: Callable[[str], str] = (lambda x: x),
    get_tracking_id_from_proteomic_sample_id: Callable[[str], str] = (lambda x: x),
) -> pd.DataFrame:
    """
    Args:
      query_result_df: pd.DataFrame containing result from get_annotation_result_from_query
      tmt_dataset: TamdemMassTagDataset
      get_tracking_id_from_proteomic_sample_id: Callable mapping proteomic sample IDs to tracking IDs
      get_tracking_id_from_genomic_sample_id: Callable mapping genomic sample IDs to tracking IDs
    """
    query_result_df = query_result_df.copy()
    proteomics_df = tmt_dataset.get_melted_abundance_df()

    query_result_df.sample_id = query_result_df.sample_id.apply(get_tracking_id_from_genomic_sample_id)
    proteomics_df.sample_id = proteomics_df.sample_id.apply(get_tracking_id_from_proteomic_sample_id)

    joined_df = query_result_df.merge(
        proteomics_df,
        left_on=["sample_id", "gene_name"],
        right_on=["sample_id", "gene_name"],
    )
    return joined_df
