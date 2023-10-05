"""Query an annotation file and return a list of sample_ids and genes meeting the query criteria."""
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from bystro.proteomics.tests.example_query import DEFAULT_FIELDS  # TK
from bystro.search.utils.messages import SaveJobData
from bystro.search.utils.opensearch import gather_opensearch_args
from bystro.utils.config import get_opensearch_config
from opensearchpy import OpenSearch

OPENSEARCH_CONFIG = get_opensearch_config()

ONE_DAY = "1d"  # default keep_alive time for opensearch point in time index


def _preprocess_query(input_query_body: dict[str, Any]) -> dict[str, Any]:
    clean_query_body = input_query_body.copy()
    clean_query_body["sort"] = ["_doc"]

    deletable_fields = ["aggs", "slice", "size"]
    for field in deletable_fields:
        clean_query_body.pop(field, None)

    return clean_query_body


SampleID = str
GeneName = str
Dosage = int


def flatten(xs: object) -> list[Any]:
    """Flatten an arbitrarily nested list."""
    if not isinstance(xs, list):
        return [xs]
    output = []
    for x in xs:
        output.extend(flatten(x))
    return output


def _process_hit(hit: dict[str, Any]) -> pd.DataFrame:
    source = hit["_source"]
    gene_names = flatten(source["refSeq"]["name2"])
    heterozygotes = flatten(source.get("heterozygotes", []))
    homozygotes = flatten(source.get("homozygotes", []))
    rows = []
    for gene_name in gene_names:
        for heterozygote in heterozygotes:
            rows.append({"sample_id": heterozygote, "gene_name": gene_name, "dosage": 1})
        for homozygote in homozygotes:
            rows.append({"sample_id": homozygote, "gene_name": gene_name, "dosage": 2})
    return pd.DataFrame(rows)


def _process_query(
    query_args: dict,
    search_client_args: dict,
) -> pd.DataFrame:
    """Process OpenSearch query and return results."""
    client = OpenSearch(**search_client_args)
    resp = client.search(**query_args)
    return _process_response(resp)


def _process_query_ray(
    query_args: dict,
    search_client_args: dict,
) -> pd.DataFrame:
    """Process OpenSearch query and return results."""
    return _process_query(query_args, search_client_args)


def _process_response(resp: dict[str, Any]) -> pd.DataFrame:
    if len(resp["hits"]["hits"]) != resp["hits"]["total"]["value"]:
        err_msg = (
            f"response hits: {len(resp['hits']['hits'])} didn't equal "
            f"total value: {resp['hits']['total']['value']}. "
            "This is a bug."
        )
        raise ValueError(err_msg)

    samples_genes_dosages_df = pd.concat([_process_hit(hit) for hit in resp["hits"]["hits"]])
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
    query_no_sort = query.copy()
    query_no_sort.pop("sort", None)  # delete these keys if present
    query_no_sort.pop("track_total_hits", None)

    response = client.count(body=query_no_sort, index=index_name)

    n_docs: int = response["count"]
    if n_docs < 1:
        err_msg = (
            f"Expected at least one document in response['count'], got response: {response} instead."
        )
        raise RuntimeError(err_msg)

    num_slices_necessary = math.ceil(n_docs / opensearch_query_options.max_query_size)
    num_slices_planned = min(num_slices_necessary, opensearch_query_options.max_slices)
    return max(num_slices_planned, 1)


def run_annotation_query(
    job_data: SaveJobData,
    search_conf: dict,
    opensearch_query_options: OpenSearchQueryOptions,
) -> pd.DataFrame:
    """Given query and index contained in SaveJobData, run query and return results in dataframe."""
    search_client_args = gather_opensearch_args(search_conf)
    client = OpenSearch(**search_client_args)

    query = job_data.queryBody
    num_slices = _get_num_slices(client, job_data.indexName, opensearch_query_options, query)
    point_in_time = client.create_point_in_time(  # type: ignore[attr-defined]
        index=job_data.indexName, params={"keep_alive": opensearch_query_options.keep_alive}
    )
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
        remote_query = _process_query(  # type: ignore[call-arg]
            query_args=query_args,
            search_client_args=search_client_args,
        )
        remote_queries.append(remote_query)
    client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore[attr-defined]
    return pd.concat(remote_queries)


def _package_opensearch_query_from_query_string(query_string: str) -> dict[str, Any]:
    return {
        "query": {
            "bool": {
                "filter": {
                    "query_string": {
                        "default_operator": "AND",
                        "query": query_string,
                        "fields": DEFAULT_FIELDS,
                        "lenient": True,
                        "phrase_slop": 5,
                        "tie_breaker": 0.3,
                    }
                }
            }
        }
    }


def get_samples_and_genes(user_query_string: str, index_name: str) -> pd.DataFrame:
    """Given a query and index, return a list of samples and genes for subsetting a proteomics matrix."""
    query = _package_opensearch_query_from_query_string(user_query_string)
    field_names = ["discordant", "homozygotes", "heterozygotes", "refSeq.name2"]
    job_data = SaveJobData(
        submissionID="1337",
        assembly="hg38",
        queryBody=_preprocess_query(query),
        indexName=index_name,
        outputBasePath=".",
        fieldNames=field_names,
    )
    samples_and_genes_df = run_annotation_query(job_data, OPENSEARCH_CONFIG, OpenSearchQueryOptions())
    return samples_and_genes_df
