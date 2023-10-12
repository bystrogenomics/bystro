"""Query an annotation file and return a list of sample_ids and genes meeting the query criteria."""
import logging
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from bystro.search.utils.messages import SaveJobData
from bystro.utils.config import get_opensearch_config
from opensearchpy import OpenSearch


logger = logging.getLogger(__file__)

OPENSEARCH_CONFIG = get_opensearch_config()
HETEROZYGOTE_DOSAGE = 1
HOMOZYGOTE_DOSAGE = 2
ONE_DAY = "1d"  # default keep_alive time for opensearch point in time index


def _preprocess_query(query_body: dict[str, Any]) -> dict[str, Any]:
    """Preprocess opensearch query by adding/updating or deleting keys."""
    clean_query_body = query_body.copy()
    clean_query_body["sort"] = ["_doc"]

    deletable_fields = ["aggs", "slice", "size"]
    for field in deletable_fields:
        clean_query_body.pop(field, None)

    return clean_query_body


def _flatten(xs: Any) -> list[Any]:
    """Flatten an arbitrarily nested list."""
    if not isinstance(xs, list):
        return [xs]
    return sum([_flatten(x) for x in xs], [])


def _get_samples_genes_dosages_from_hit(hit: dict[str, Any]) -> pd.DataFrame:
    """Given a document hit, return a dataframe of samples, genes and dosages."""
    source = hit["_source"]
    gene_names = _flatten(source["refSeq"]["name2"])
    # homozygotes, heterozygotes may not be present in response, so
    # represent them as empty lists if not.
    heterozygotes = _flatten(source.get("heterozygotes", []))
    homozygotes = _flatten(source.get("homozygotes", []))
    rows = []
    for gene_name in gene_names:
        for heterozygote in heterozygotes:
            rows.append(
                {"sample_id": heterozygote, "gene_name": gene_name, "dosage": HETEROZYGOTE_DOSAGE}
            )
        for homozygote in homozygotes:
            rows.append({"sample_id": homozygote, "gene_name": gene_name, "dosage": HOMOZYGOTE_DOSAGE})
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


@dataclass
class OpenSearchQueryOptions:
    """Represent parameters for configuring OpenSearch queries."""

    max_query_size: int = 10_000
    max_slices: int = 1024
    keep_alive: str = ONE_DAY


def _get_num_slices(
    client: OpenSearch,
    index_name: str,
    opensearch_query_options: OpenSearchQueryOptions,
    query: dict[str, Any],
) -> int:
    """Count number of hits for the index."""
    get_num_slices_query = query.copy()
    get_num_slices_query.pop("sort", None)
    get_num_slices_query.pop("track_total_hits", None)

    response = client.count(body=get_num_slices_query, index=index_name)

    n_docs: int = response["count"]
    if n_docs < 1:
        err_msg = (
            f"Expected at least one document in `response['count']`, got response: {response} instead."
        )
        raise RuntimeError(err_msg)

    num_slices_necessary = math.ceil(n_docs / opensearch_query_options.max_query_size)
    num_slices_planned = min(num_slices_necessary, opensearch_query_options.max_slices)
    return max(num_slices_planned, 1)


def _run_annotation_query(
    job_data: SaveJobData,
    client: OpenSearch,
    opensearch_query_options: OpenSearchQueryOptions,
) -> pd.DataFrame:
    """Given query and index contained in SaveJobData, run query and return  in dataframe."""

    query = job_data.queryBody
    num_slices = _get_num_slices(client, job_data.indexName, opensearch_query_options, query)
    point_in_time = client.create_point_in_time(  # type: ignore[attr-defined]
        index=job_data.indexName, params={"keep_alive": opensearch_query_options.keep_alive}
    )
    try:  # make sure we clean up the PIT index properly no matter what happens in this block
        pit_id = point_in_time["pit_id"]
        query["pit"] = {"id": pit_id}
        query["size"] = opensearch_query_options.max_query_size
        remote_queries = []
        for slice_id in range(num_slices):
            slice_query = query.copy()
            if num_slices > 1:
                # Slice queries require max > 1
                slice_query["slice"] = {"id": slice_id, "max": num_slices}
            query_args = {"body": slice_query}
            query_result = _execute_query(  # type: ignore[call-arg]
                client,
                query_args=query_args,
            )
            remote_queries.append(query_result)
    except Exception as e:
        err_msg = (
            f"Encountered exception: {repr(e)} while running opensearch_query, "
            "deleting PIT index and exiting.\n"
            f"job_data: {job_data}\n"
            f"client: {client}\n"
            f"opensearch_query_options: {opensearch_query_options}\n"
        )
        logger.exception(err_msg, exc_info=e)
        client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore[attr-defined]
        raise
    client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore[attr-defined]
    return pd.concat(remote_queries)


def _build_opensearch_query_from_query_string(query_string: str) -> dict[str, Any]:
    return {
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
    }


def get_samples_and_genes_from_query(
    user_query_string: str,
    index_name: str,
    client: OpenSearch,
) -> pd.DataFrame:
    """Given a query and index, return a dataframe of (sample_id, gene, dosage) rows matching query."""
    query = _build_opensearch_query_from_query_string(user_query_string)
    job_data = SaveJobData(
        submissionID="1337",
        assembly="hg38",
        queryBody=_preprocess_query(query),
        indexName=index_name,
        outputBasePath=".",
        fieldNames=[],
    )
    samples_and_genes_df = _run_annotation_query(job_data, client, OpenSearchQueryOptions())
    return samples_and_genes_df
