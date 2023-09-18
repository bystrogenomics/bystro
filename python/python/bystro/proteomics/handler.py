"""A module to handle the saving of search results into a new annotation"""
from typing import Tuple
from typing import Optional
import gzip
import math
import os
import pathlib
import subprocess
import traceback
from typing import Any

import numpy as np
import ray

from opensearchpy import OpenSearch

from bystro.beanstalkd.worker import ProgressPublisher, get_progress_reporter
from bystro.search.utils.annotation import AnnotationOutputs, get_delimiters
from bystro.search.utils.messages import SaveJobData
from bystro.search.utils.opensearch import gather_opensearch_args
from bystro.proteomics.utils import GNU_TAR_EXECUTABLE_NAME

ray.init(ignore_reinit_error=True, address="auto")
RAY_ERROR = -1  # the error code we use if a ray remote job fails to return

ONE_DAY = "1d"  # default keep_alive time for opensearch point in time index


def _preprocess_query(input_query_body: dict[str, Any]) -> dict[str, Any]:
    clean_query_body = input_query_body.copy()
    clean_query_body["sort"] = ["_doc"]

    deletable_fields = ["aggs", "slice", "size"]
    for field in deletable_fields:
        clean_query_body.pop("aggs", None)

    return clean_query_body


def _get_header(field_names: list[str]) -> Tuple[list[str], list[str | list[str]]]:
    parents: list[str] = []
    children: list[str | list[str]] = []
    for i, field in enumerate(field_names):
        if "." in field:
            path = field.split(".")
            parents.append(path[0])
            children.append(path[1:])
        else:
            parents.append(field)
            children.append(field)

    return parents, children


def _populate_data(
    field_path: list[str] | str, data_for_end_of_path: dict[str, Any] | None
) -> Optional[dict[str, Any]]:
    if not isinstance(field_path, list) or data_for_end_of_path is None:
        return data_for_end_of_path

    for child_field in field_path:
        data_for_end_of_path = data_for_end_of_path.get(child_field)

        if data_for_end_of_path is None:
            return data_for_end_of_path

    return data_for_end_of_path


def _make_output_string(rows: list[list[None | str]], delims: dict[str, str]) -> bytes:
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


def _process_rows(resp, field_names):
    parent_fields, child_fields = _get_header(field_names)
    try:
        discordant_idx = field_names.index("discordant")
    except ValueError:
        err_msg = f"response: {resp} with field_names: {field_names} lacked field: 'discordant'"
        raise ValueError(err_msg)
    rows = []
    for doc in resp["hits"]["hits"]:
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
    reporter,
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
    try:
        with gzip.open(chunk_output_name, "wb") as fw:
            fw.write(_make_output_string(rows, delimiters))
        reporter.increment.remote(resp["hits"]["total"]["value"])
    except Exception:
        traceback.print_exc()
        return RAY_ERROR

    return resp["hits"]["total"]["value"]


def _get_num_slices(client, index_name, max_query_size, max_slices, query) -> int:
    """Count number of hits for the index"""
    query_no_sort = query.copy()
    query_no_sort.pop("sort", None)  # delete these keys if present
    query_no_sort.pop("track_total_hits", None)

    response = client.count(body=query_no_sort, index=index_name)

    n_docs = response["count"]
    assert n_docs > 0

    num_slices_necessary = math.ceil(n_docs / max_query_size)
    num_slices_planned = min(num_slices_necessary, max_slices)
    return max(num_slices_planned, 1)


def run_query_and_write_output(
    job_data: SaveJobData,
    search_conf: dict,
    publisher: ProgressPublisher,
    max_query_size: int = 10_000,
    max_slices=1024,
    keep_alive=ONE_DAY,
) -> AnnotationOutputs:
    """Main function for running the query and writing the output"""
    output_dir = os.path.dirname(job_data.outputBasePath)
    basename = os.path.basename(job_data.outputBasePath)
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
    reporter = get_progress_reporter(publisher)
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
                reporter=reporter,
                delimiters=get_delimiters(),
            )
            remote_queries.append(remote_query)
        results_processed = ray.get(remote_queries)

        if RAY_ERROR in results_processed:
            raise IOError("Failed to process chunk")

        _write_tarball(written_chunks, output_dir, outputs)
    except IOError as io_err:
        client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore
        raise IOError(io_err) from io_er

    client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore

    return outputs


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
