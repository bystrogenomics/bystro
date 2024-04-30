"""Query an annotation file and return a list of sample_ids and genes meeting the query criteria."""
import logging
import math
from typing import Any, Callable

from msgspec import Struct
import numpy as np
import pandas as pd
from opensearchpy import OpenSearch

from bystro.proteomics.fragpipe_tandem_mass_tag import TandemMassTagDataset

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


def _extract_samples(samples):
    return [sample[0] for sample in samples]


def _get_samples_genes_dosages_from_hit(hit: dict[str, Any]) -> pd.DataFrame:
    """Given a document hit, return a dataframe of samples, genes and dosages."""
    source = hit["_source"]
    chrom = _flatten(source["chrom"])[0]
    pos = _flatten(source["pos"])[0]
    ref = _flatten(source["ref"])[0]
    alt = _flatten(source["alt"])[0]
    unique_gene_names = set(_flatten(source["refSeq"]["name2"]))
    # homozygotes, heterozygotes may not be present in response, so
    # represent them as empty lists if not.

    heterozygotes = _flatten(source.get("heterozygotes", []))
    homozygotes = _flatten(source.get("homozygotes", []))
    missing_genos = _flatten(source.get("missingGenos", []))
    rows = []
    for gene_name in unique_gene_names:
        for heterozygote in heterozygotes:
            rows.append(
                {
                    "sample_id": heterozygote,
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "gene_name": gene_name,
                    "dosage": HETEROZYGOTE_DOSAGE,
                }
            )
        for homozygote in homozygotes:
            rows.append(
                {
                    "sample_id": homozygote,
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "gene_name": gene_name,
                    "dosage": HOMOZYGOTE_DOSAGE,
                }
            )
        for missing_geno in missing_genos:
            rows.append(
                {
                    "sample_id": missing_geno,
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "gene_name": gene_name,
                    "dosage": MISSING_GENO_DOSAGE,
                }
            )

    return pd.DataFrame(rows)


def _execute_query(
    client: OpenSearch,
    query_args: dict,
) -> pd.DataFrame:
    """Process OpenSearch query and return results."""
    resp = client.search(**query_args)
    return _process_response(resp)


def _process_response(resp: dict[str, Any]) -> pd.DataFrame:
    """Postprocess query response from opensearch client."""
    num_hits = len(resp["hits"]["hits"])
    total_value = resp["hits"]["total"]["value"]
    if num_hits != total_value:
        err_msg = f"Number of hits: {num_hits} didn't equal total value: {total_value}. This is a bug."
        raise ValueError(err_msg)

    samples_genes_dosages_df = pd.concat(
        [_get_samples_genes_dosages_from_hit(hit) for hit in resp["hits"]["hits"]]
    )
    # we may have multiple variants per gene in the results, so we
    # need to drop duplicates here.
    return samples_genes_dosages_df.drop_duplicates()


def _get_num_slices(
    client: OpenSearch,
    index_name: str,
    query: dict[str, Any],
) -> tuple[int, int]:
    """Count number of hits for the index."""
    get_num_slices_query = query["body"].copy()
    get_num_slices_query.pop("sort", None)
    get_num_slices_query.pop("track_total_hits", None)

    response = client.count(body=get_num_slices_query, index=index_name)

    n_docs: int = response["count"]
    if n_docs < 1:
        err_msg = (
            f"Expected at least one document in `response['count']`, got response: {response} instead."
        )
        raise RuntimeError(err_msg)

    num_slices_necessary = math.ceil(n_docs / OPENSEARCH_QUERY_CONFIG.max_query_size)
    num_slices_planned = min(num_slices_necessary, OPENSEARCH_QUERY_CONFIG.max_slices)
    return max(num_slices_planned, 1), n_docs


def _run_annotation_query(
    query: dict[str, Any],
    index_name: str,
    client: OpenSearch,
) -> pd.DataFrame:
    """Given query and index contained in SaveJobData, run query and return results as dataframe."""
    num_slices, _ = _get_num_slices(client, index_name, query)
    point_in_time = client.create_point_in_time(  # type: ignore[attr-defined]
        index=index_name, params={"keep_alive": OPENSEARCH_QUERY_CONFIG.keep_alive}
    )
    try:  # make sure we clean up the PIT index properly no matter what happens in this block
        pit_id = point_in_time["pit_id"]
        query["body"]["pit"] = {"id": pit_id}
        query["body"]["size"] = OPENSEARCH_QUERY_CONFIG.max_query_size
        query_results = []
        for slice_id in range(num_slices):
            slice_query = query.copy()
            if num_slices > 1:
                # Slice queries require max > 1
                slice_query["slice"] = {"id": slice_id, "max": num_slices}
            query_result = _execute_query(
                client,
                query_args=query,
            )
            query_results.append(query_result)
    except Exception as e:
        err_msg = (
            f"Encountered exception: {e!r} while running opensearch_query, "
            "deleting PIT index and exiting.\n"
            f"query: {query}\n"
            f"client: {client}\n"
            f"opensearch_query_config: {OPENSEARCH_QUERY_CONFIG}\n"
        )
        logger.exception(err_msg, exc_info=e)
        client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore[attr-defined]
        raise
    client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore[attr-defined]
    return pd.concat(query_results)


def _build_opensearch_query_from_query_string(query_string: str) -> dict[str, Any]:
    base_query = {
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
        },
        "_source_includes": OUTPUT_FIELDS,
    }
    return base_query


def get_annotation_result_from_query(
    user_query_string: str,
    index_name: str,
    client: OpenSearch,
) -> pd.DataFrame:
    """Given a query and index, return a dataframe of variant / sample_id records matching query."""
    query = _build_opensearch_query_from_query_string(user_query_string)
    return _run_annotation_query(query, index_name, client)


def join_annotation_result_to_proteomics_dataset(
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
