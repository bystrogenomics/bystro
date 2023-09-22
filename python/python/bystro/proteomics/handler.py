from typing import Tuple

import gzip
import math
import os
import pathlib
import subprocess
import traceback
from typing import Any
from pathlib import Path
import numpy as np
import ray
import pandas as pd
from opensearchpy import OpenSearch
import io
from bystro.beanstalkd.worker import ProgressPublisher
from bystro.search.utils.annotation import AnnotationOutputs, get_delimiters
from bystro.search.utils.messages import SaveJobData
from bystro.search.utils.opensearch import gather_opensearch_args
from bystro.proteomics.utils import GNU_TAR_EXECUTABLE_NAME
from bystro.proteomics.tests.example_query import DEFAULT_FIELDS  # TK
from ruamel.yaml import YAML

ray.init(ignore_reinit_error=True, address="auto")
RAY_ERROR = -1  # the error code we use if a ray remote job fails to return

ONE_DAY = "1d"  # default keep_alive time for opensearch point in time index


with open(Path.home() / "bystro/config/opensearch.yml") as search_config_file:
    SEARCH_CONF = YAML(typ="safe").load(search_config_file)
    SEARCH_CONF["connection"]["request_timeout"] = 5


def _preprocess_query(input_query_body: dict[str, Any]) -> dict[str, Any]:
    clean_query_body = input_query_body.copy()
    clean_query_body["sort"] = ["_doc"]

    deletable_fields = ["aggs", "slice", "size"]
    for field in deletable_fields:
        clean_query_body.pop(field, None)

    return clean_query_body


def _get_header(field_names: list[str]) -> Tuple[list[str], list[str | list[str]]]:
    parents: list[str] = []
    children: list[str | list[str]] = []
    for field in field_names:
        if "." in field:
            path = field.split(".")
            parents.append(path[0])
            children.append(path[1:])
        else:
            parents.append(field)
            children.append(field)

    return parents, children


def _populate_data(
    field_path: list[str] | str, data_for_end_of_path: list | dict[str, Any] | None
) -> dict[str, Any] | list[str] | None:
    if not isinstance(field_path, list) or data_for_end_of_path is None:
        return data_for_end_of_path
    assert isinstance(data_for_end_of_path, dict)
    for child_field in field_path:
        data_for_end_of_path = data_for_end_of_path.get(child_field)
        if data_for_end_of_path is None:
            return data_for_end_of_path

    return data_for_end_of_path


def _make_output_string(rows: list[list[list[list[str | None]]]], delims: dict[str, str]) -> bytes:
    empty_field_char = delims["empty_field"]
    for row_idx, row in enumerate(rows):  # pylint:disable=too-many-nested-blocks
        # Some fields may just be missing; we won't store even the alt/pos [[]] structure for those
        for i, column in enumerate(row):
            if column is None:
                row[i] = empty_field_char
                continue

            # For now, we don't store multiallelics; top level array is placeholder only
            # With breadth 1
            if not isinstance(column, list):
                row[i] = str(column)
                continue

            for j, position_data in enumerate(column):
                if position_data is None:
                    column[j] = empty_field_char
                    continue

                if isinstance(position_data, list):
                    inner_values = []
                    for sub in position_data:
                        if sub is None:
                            inner_values.append(empty_field_char)
                            continue

                        if isinstance(sub, list):
                            inner_values.append(
                                delims["value"].join(
                                    map(lambda x: str(x) if x is not None else empty_field_char, sub)
                                )
                            )
                        else:
                            inner_values.append(str(sub))

                    column[j] = delims["position"].join(inner_values)

            row[i] = delims["overlap"].join(column)

        rows[row_idx] = delims["field"].join(row)

    return bytes("\n".join(rows) + "\n", encoding="utf-8")


def _make_output_string_spec(rows, delims):
    empty_field_char = delims["empty_field"]
    for row_idx, row in enumerate(rows):  # pylint:disable=too-many-nested-blocks
        # Some fields may just be missing; we won't store even the alt/pos [[]] structure for those
        row = do_row(row)
        rows[row_idx] = row
    return bytes("\n".join(rows) + "\n", encoding="utf-8")


import copy


def do_row(input_row):
    row = copy.deepcopy(input_row)
    delims = get_delimiters()
    empty_field_char = delims["empty_field"]
    for i, column in enumerate(row):
        if column is None:
            row[i] = empty_field_char
            continue

        # For now, we don't store multiallelics; top level array is placeholder only
        # With breadth 1
        if not isinstance(column, list):
            row[i] = str(column)
            continue

        column = do_column(column)
        row[i] = column
    return delims["field"].join(row)


def do_column(input_column):
    column = copy.deepcopy(input_column)
    delims = get_delimiters()
    empty_field_char = delims["empty_field"]

    for j, position_data in enumerate(column):
        if position_data is None:
            column[j] = empty_field_char
            continue

        if isinstance(position_data, list):
            inner_values = do_position_data(position_data)
            column[j] = inner_values
    column = delims["overlap"].join(column)
    return column


def do_position_data(input_position_data):
    position_data = copy.deepcopy(input_position_data)
    delims = get_delimiters()
    empty_field_char = delims["empty_field"]

    inner_values = []
    for sub in position_data:
        if sub is None:
            inner_values.append(empty_field_char)
            continue
        inner_values.append(do_sub(sub))
    return delims["position"].join(inner_values)


def do_sub(sub):
    delims = get_delimiters()
    empty_field_char = delims["empty_field"]
    if isinstance(sub, list):
        return delims["value"].join(map(to_string_or_empty_field_char, sub))
    else:
        return str(sub)


def to_string_or_empty_field_char(x):
    delims = get_delimiters()
    empty_field_char = delims["empty_field"]

    return str(x) if x is not None else empty_field_char


def _process_rows(resp, field_names):
    with open("safe_response.py", "w") as f:
        f.write(str(resp))
    parent_fields, child_fields = _get_header(field_names)
    try:
        discordant_idx = field_names.index("discordant")
    except ValueError:
        err_msg = f"response: {resp} with field_names: {field_names} lacked field: 'discordant'"
        raise ValueError(err_msg)
    rows = []
    docs = resp["hits"]["hits"]
    for doc in docs:
        row = np.empty(len(field_names), dtype=object)
        for field_name_idx in range(len(field_names)):
            parent_field = parent_fields[field_name_idx]
            row[field_name_idx] = _populate_data(
                child_fields[field_name_idx], doc["_source"].get(parent_field)
            )
        if row[discordant_idx][0][0] is False:
            row[discordant_idx][0][0] = 0
        elif row[discordant_idx][0][0] is True:
            row[discordant_idx][0][0] = 1
        rows.append(row)
    return rows


@ray.remote
def _process_query(
    query_args: dict,
    search_client_args: dict,
    field_names: list,
    chunk_output_name: str,
    delimiters,
):
    """Process OpenSearch query and save results.

    - Run OpenSearch query against index specified in
    - search_client_args Write output to disk under chunk_output_name.
    - Warning: this method does not implement exception-based error
      handling but returns an error code of RAY_ERROR if any exception is encountered.
    - returns... what? TK
    """
    client = OpenSearch(**search_client_args)
    resp = client.search(**query_args)

    if resp["hits"]["total"]["value"] == 0:
        return 0

    # Each sliced scroll chunk should get all records for that chunk
    assert len(resp["hits"]["hits"]) == resp["hits"]["total"]["value"]

    rows = _process_rows(resp, field_names)
    output_string = _make_output_string(rows, delimiters)
    try:
        with gzip.open(chunk_output_name, "wb") as fw:
            fw.write(output_string)
        resp["hits"]["total"]["value"] += 1
    except Exception:
        traceback.print_exc()
        return RAY_ERROR

    return resp["hits"]["total"]["value"]


@ray.remote
def _process_query_pure(
    query_args: dict,
    search_client_args: dict,
    field_names: list,
    delimiters,
):
    """Process OpenSearch query and return results."""
    client = OpenSearch(**search_client_args)
    resp = client.search(**query_args)

    if resp["hits"]["total"]["value"] == 0:
        return 0

    # Each sliced scroll chunk should get all records for that chunk
    assert len(resp["hits"]["hits"]) == resp["hits"]["total"]["value"]

    rows = _process_rows(resp, field_names)
    print("rows:", rows)
    output_string = _make_output_string(rows, delimiters)
    return output_string


def _get_num_slices(client, index_name, max_query_size, max_slices, query) -> int:
    """Count number of hits for the index"""
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


def run_query_and_write_output(
    job_data: SaveJobData,
    search_conf: dict,
    max_query_size: int = 10_000,
    max_slices=1024,
    keep_alive=ONE_DAY,
) -> AnnotationOutputs:
    """Main function for running the query and writing the output"""
    output_dir = Path(job_data.outputBasePath).parent.absolute()
    basename = Path(job_data.outputBasePath).name
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    outputs = AnnotationOutputs.from_path(output_dir, basename, compress=True)

    written_chunks = [os.path.join(output_dir, f"{job_data.indexName}_header")]

    header = bytes("\t".join(job_data.fieldNames) + "\n", encoding="utf-8")
    with gzip.open(written_chunks[0], "wb") as fw:
        fw.write(header)  # type: ignore

    search_client_args = gather_opensearch_args(search_conf)
    client = OpenSearch(**search_client_args)

    query = _preprocess_query(job_data.queryBody)
    num_slices = _get_num_slices(client, job_data.indexName, max_query_size, max_slices, query)
    point_in_time = client.create_point_in_time(  # type: ignore[attr-defined]
        index=job_data.indexName, params={"keep_alive": keep_alive}
    )
    pit_id = point_in_time["pit_id"]
    query["pit"] = {"id": pit_id}
    query["size"] = max_query_size
    remote_queries = []
    try:
        for slice_id in range(num_slices):
            chunk_output_name = os.path.join(output_dir, f"{job_data.indexName}_{slice_id}")
            written_chunks.append(chunk_output_name)
            slice_query = query.copy()
            if num_slices > 1:
                # Slice queries require max > 1
                slice_query["slice"] = {"id": slice_id, "max": num_slices}
            remote_query = _process_query.remote(
                query_args={"body": slice_query},
                search_client_args=search_client_args,
                field_names=job_data.fieldNames,
                chunk_output_name=chunk_output_name,
                delimiters=get_delimiters(),
            )
            remote_queries.append(remote_query)
        results_processed = ray.get(remote_queries)

        if RAY_ERROR in results_processed:
            raise IOError("Failed to process chunk")

        _write_tarball(written_chunks, output_dir, outputs)
    except IOError as io_err:
        client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore
        raise IOError(io_err) from io_err

    client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore

    return outputs


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
    try:
        for slice_id in range(num_slices):
            slice_query = query.copy()
            if num_slices > 1:
                # Slice queries require max > 1
                slice_query["slice"] = {"id": slice_id, "max": num_slices}
            remote_query = _process_query_pure.remote(
                query_args={"body": slice_query},
                search_client_args=search_client_args,
                field_names=job_data.fieldNames,
                delimiters=delimiters,
            )
            remote_queries.append(remote_query)
        results_processed = ray.get(remote_queries)

        if RAY_ERROR in results_processed:
            raise IOError("Failed to process chunk")
    except IOError as io_err:
        client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore
        raise IOError(io_err) from io_err

    client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore

    payload = [header] + results_processed
    payload_str = "".join([byte_str.decode("utf-8") for byte_str in payload])
    samples_and_genes_df = pd.read_csv(io.StringIO(payload_str), sep="\t")
    assert all(
        samples_and_genes_df.columns == ["discordant", "homozygotes", "heterozygotes", "refSeq.name2"]
    )
    samples_and_genes_df = samples_and_genes_df.drop("discordant", axis=1)
    samples_and_genes_df = samples_and_genes_df.rename({"refSeq.name2": "gene_name"}, axis=1)
    output = []
    for i, row in samples_and_genes_df.iterrows():
        gene_names = deduplicate(split_gene_names(row.gene_name, delimiters))
        homozygotes = row.homozygotes.split("|") if row.homozygotes != "!" else []
        heterozygotes = row.heterozygotes.split("|") if row.heterozygotes != "!" else []
        for gene_name in gene_names:
            for heterozygote in heterozygotes:
                output.append((heterozygote, gene_name, 1))
            for homozygote in homozygotes:
                output.append((homozygote, gene_name, 2))
    return pd.DataFrame(output, columns=["sample_id", "gene_name", "dosage"])


def split_gene_names(gene_names: str, delimiters: dict[str, str]) -> list[str]:
    names = []
    for group in gene_names.split(delimiters["position"]):
        for name in group.split(delimiters["overlap"]):
            names.append(name)
    return names


def deduplicate(xs: list) -> list:
    return list(set(xs))


def _write_tarball(written_chunks: list[str], output_dir: str, outputs: AnnotationOutputs) -> None:
    all_chunks = " ".join(written_chunks)
    annotation_path = os.path.join(output_dir, outputs.annotation)
    cat_command = f"cat {all_chunks} > {annotation_path}; rm {all_chunks}"
    ret_code = subprocess.call(cat_command, shell=True)
    if ret_code != 0:
        raise IOError(f"Failed to write {annotation_path}")

    tarball_name = os.path.basename(outputs.archived)
    tar_command = (
        f"cd {output_dir} && "
        f'{GNU_TAR_EXECUTABLE_NAME} --exclude ".*" --exclude={tarball_name} '
        f"-cf {tarball_name} * --remove-files"
    )
    ret_code = subprocess.call(tar_command, shell=True)
    if ret_code != 0:
        raise IOError(
            f"Command: '{tar_command}', failed with retcode {ret_code}, "
            f"could not generate: {outputs.archived}"
        )


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


def get_samples_and_genes(user_query_string: str, index_name: str) -> Tuple[set[str], set[str]]:
    """Given a query and index, return a list of samples and genes for subsetting a proteomics matrix."""
    query = _package_opensearch_query_from_query_string(user_query_string)
    field_names = ["discordant", "homozygotes", "heterozygotes", "refSeq.name2"]
    job_data = SaveJobData(
        submissionID="1337",
        assembly="hg38",
        queryBody=query,
        indexName=index_name,
        outputBasePath=None,
        fieldNames=field_names,
    )
    publisher = ProgressPublisher(host="127.0.0.1", port=1337, queue="proteomics", message=None)
    samples_and_genes_df = run_query_and_write_output_pure(job_data, SEARCH_CONF)
    samples = set(samples_and_genes_df.sample_id)
    genes = set(samples_and_genes_df.gene_name)
    return samples, genes
