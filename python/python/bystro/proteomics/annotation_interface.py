"""Query an annotation file and return a list of sample_ids and genes meeting the query criteria."""
import io
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
import ray
from bystro.proteomics.tests.example_query import DEFAULT_FIELDS, SPEC_FIELDS  # TK
from bystro.search.utils.annotation import AnnotationOutputs, get_delimiters
from bystro.search.utils.messages import SaveJobData
from bystro.search.utils.opensearch import gather_opensearch_args
from bystro.utils.config import get_opensearch_config
from opensearchpy import OpenSearch
from ruamel.yaml import YAML

ray.init(ignore_reinit_error=True, address="auto")
OPENSEARCH_CONFIG = get_opensearch_config()
RAY_ERROR = -1  # the error code we use if a ray remote job fails to return

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


def flatten(xs):
    if not isinstance(xs, list):
        return [xs]
    output = []
    for x in xs:
        output.extend(flatten(x))
    return output


def _process_hit(hit: dict[str, Any]) -> list[tuple[SampleID, GeneName, Dosage]]:
    source = hit["_source"]
    gene_names = flatten(source["refSeq"]["name2"])
    heterozygotes = flatten(source.get("heterozygotes", []))
    homozygotes = flatten(source.get("homozygotes", []))
    rows = []
    for gene_name in gene_names:
        for heterozygote in heterozygotes:
            rows.append(({"sample_id": heterozygote, "gene_name": gene_name, "dosage": 1}))
        for homozygote in homozygotes:
            rows.append(({"sample_id": homozygote, "gene_name": gene_name, "dosage": 2}))
    return pd.DataFrame(rows)


@ray.remote
def _process_query_pure(
    query_args: dict,
    search_client_args: dict,
    field_names: list,
    delimiters: dict[str, str],
) -> pd.DataFrame:
    """Process OpenSearch query and return results."""
    client = OpenSearch(**search_client_args)
    resp = client.search(**query_args)
    return _process_response(resp)


def _process_response(resp) -> pd.DataFrame:
    if len(resp["hits"]["hits"]) != resp["hits"]["total"]["value"]:
        err_msg = (
            f"response hits: {len(resp['hits']['hits'])} didn't equal "
            f"total value: {resp['hits']['total']['value']}. "
            "This is a bug."
        )
        raise ValueError(err_msg)

    df = pd.concat([_process_hit(hit) for hit in resp["hits"]["hits"]])
    # we may have multiple variants per gene in the results, so we
    # need to drop duplicates here.
    return df.drop_duplicates()


def _get_num_slices(
    client: OpenSearch,
    index_name: str,
    max_query_size: int,
    max_slices: int,
    query: dict[str, Any],
) -> int:
    """Count number of hits for the index."""
    query_no_sort = query.copy()
    query_no_sort.pop("sort", None)  # delete these keys if present
    query_no_sort.pop("track_total_hits", None)

    response = client.count(body=query_no_sort, index=index_name)

    n_docs = response["count"]
    if n_docs < 1:
        err_msg = (
            f"Expected at least one document in response['count'], got response: {response} instead."
        )
        raise RuntimeError(err_msg)

    num_slices_necessary = math.ceil(n_docs / max_query_size)
    num_slices_planned = min(num_slices_necessary, max_slices)
    return max(num_slices_planned, 1)


def run_query_and_write_output_pure(
    job_data: SaveJobData,
    search_conf: dict,
    max_query_size: int = 10_000,
    max_slices=1024,
    keep_alive=ONE_DAY,
) -> pd.DataFrame:
    assert isinstance(max_query_size, int), type(max_query_size)
    header = bytes("\t".join(job_data.fieldNames) + "\n", encoding="utf-8")

    search_client_args = gather_opensearch_args(search_conf)
    client = OpenSearch(**search_client_args)
    delimiters = get_delimiters()

    query = _preprocess_query(job_data.queryBody)
    num_slices = _get_num_slices(client, job_data.indexName, max_query_size, max_slices, query)
    point_in_time = client.create_point_in_time(  # type: ignore[attr-defined]
        index=job_data.indexName, params={"keep_alive": keep_alive}
    )
    pit_id = point_in_time["pit_id"]
    query["pit"] = {"id": pit_id}
    query["size"] = max_query_size
    remote_queries = []
    for slice_id in range(num_slices):
        slice_query = query.copy()
        if num_slices > 1:
            # Slice queries require max > 1
            slice_query["slice"] = {"id": slice_id, "max": num_slices}
        remote_query = _process_query_pure.remote(  # type: ignore[call-arg]
            query_args={"body": slice_query},
            search_client_args=search_client_args,
            field_names=job_data.fieldNames,
            delimiters=delimiters,
        )
        remote_queries.append(remote_query)
    results_processed = ray.get(remote_queries)
    client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore
    return pd.concat(results_processed)


def _package_opensearch_query_from_query_string(query_string: str) -> dict[str, Any]:
    return {
        "query": {
            "bool": {
                "filter": {
                    "query_string": {
                        "default_operator": "AND",
                        "query": query_string,
                        # "fields": SPEC_FIELDS,
                        "fields": DEFAULT_FIELDS,
                        "lenient": True,
                        "phrase_slop": 5,
                        "tie_breaker": 0.3,
                    }
                }
            }
        }
    }


def get_samples_and_genes(user_query_string: str, index_name: str) -> Tuple[set[str], set[str]]:
    """Given a query and index, return a list of samples and genes for subsetting a proteomics matrix."""
    query = _package_opensearch_query_from_query_string(user_query_string)
    field_names = ["discordant", "homozygotes", "heterozygotes", "refSeq.name2"]
    job_data = SaveJobData(
        submissionID="1337",
        assembly="hg38",
        queryBody=query,
        indexName=index_name,
        outputBasePath=".",
        fieldNames=field_names,
    )
    samples_and_genes_df = run_query_and_write_output_pure(job_data, OPENSEARCH_CONFIG)
    return samples_and_genes_df
