"""A module to handle the saving of search results into a new annotation"""
# TODO 2023-05-08: Track number of skipped entries
# TODO 2023-05-08: Write to Arrow IPC or Parquet, alongside tsv.
# TODO 2023-05-08: Implement distributed pipeline/filters/transforms
# TODO 2023-05-08: Support sort queries
# TODO 2023-05-08: get max_slices from opensearch index settings
# TODO 2023-05-08: concatenate chunks in a different ray worker

import math
import os
import pathlib
import subprocess
import traceback

import numpy as np
import ray
from isal import igzip
from opensearchpy import OpenSearch

from bystro.beanstalkd.worker import ProgressPublisher, get_progress_reporter
from bystro.search.utils.annotation import AnnotationOutputs, get_delimiters
from bystro.search.utils.messages import SaveJobData
from bystro.search.utils.opensearch import gather_opensearch_args

ray.init(ignore_reinit_error=True, address="auto")


def _clean_query(input_query_body: dict):
    input_query_body["sort"] = ["_doc"]

    if "aggs" in input_query_body:
        del input_query_body["aggs"]

    if "slice" in input_query_body:
        del input_query_body["slice"]

    if "size" in input_query_body:
        del input_query_body["size"]

    return input_query_body


def _get_header(field_names):
    children = [None] * len(field_names)
    parents = [None] * len(field_names)

    for i, field in enumerate(field_names):
        if "." in field:
            path = field.split(".")
            parents[i] = path[0]
            children[i] = path[1:]
        else:
            parents[i] = field
            children[i] = field

    return parents, children


def _populate_data(field_path, data_for_end_of_path):
    if not isinstance(field_path, list):
        return data_for_end_of_path

    for child_field in field_path:
        data_for_end_of_path = data_for_end_of_path[child_field]

    return data_for_end_of_path


def _make_output_string(rows: list, delims: dict):
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


@ray.remote
def _process_query(
    query_args: dict,
    search_client_args: dict,
    field_names: list,
    chunk_output_name: str,
    reporter,
    delimiters,
):
    client = OpenSearch(**search_client_args)
    resp = client.search(**query_args)

    if resp["hits"]["total"]["value"] == 0:
        return 0

    rows = []

    parent_fields, child_fields = _get_header(field_names)

    discordant_idx = field_names.index("discordant")
    assert discordant_idx > -1

    # Each sliced scroll chunk should get all records for that chunk
    assert len(resp["hits"]["hits"]) == resp["hits"]["total"]["value"]

    for doc in resp["hits"]["hits"]:
        row = np.empty(len(field_names), dtype=object)
        for y in range(len(field_names)):
            row[y] = _populate_data(child_fields[y], doc["_source"][parent_fields[y]])

        if row[discordant_idx][0][0] is False:
            row[discordant_idx][0][0] = 0
        elif row[discordant_idx][0][0] is True:
            row[discordant_idx][0][0] = 1

        rows.append(row)

    try:
        with igzip.open(chunk_output_name, "wb") as fw:
            fw.write(_make_output_string(rows, delimiters))  # type: ignore
        reporter.increment.remote(resp["hits"]["total"]["value"])
    except Exception:
        traceback.print_exc()
        return -1

    return resp["hits"]["total"]["value"]


def _get_num_slices(client, index_name, max_query_size, max_slices, query):
    """Count number of hits for the index"""
    query_no_sort = query.copy()
    if "sort" in query_no_sort:
        del query_no_sort["sort"]
    if "track_total_hits" in query_no_sort:
        del query_no_sort["track_total_hits"]

    response = client.count(body=query_no_sort, index=index_name)

    n_docs = response["count"]
    assert n_docs > 0

    return max(min(math.ceil(n_docs / max_query_size), max_slices), 1)


async def go(  # pylint:disable=invalid-name
    job_data: SaveJobData,
    search_conf: dict,
    publisher: ProgressPublisher,
    max_query_size: int = 10_000,
    max_slices=1024,
    keep_alive="1d",
) -> AnnotationOutputs:
    """Main function for running the query and writing the output"""
    output_dir = os.path.dirname(job_data.outputBasePath)
    basename = os.path.basename(job_data.outputBasePath)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    outputs = AnnotationOutputs.from_path(output_dir, basename, True)

    written_chunks = [os.path.join(output_dir, f"{job_data.indexName}_header")]

    header = bytes("\t".join(job_data.fieldNames) + "\n", encoding="utf-8")
    with igzip.open(written_chunks[-1], "wb") as fw:
        fw.write(header)  # type: ignore

    search_client_args = gather_opensearch_args(search_conf)
    client = OpenSearch(**search_client_args)

    query = _clean_query(job_data.queryBody)
    num_slices = _get_num_slices(client, job_data.indexName, max_query_size, max_slices, query)
    pit_id = client.create_point_in_time(index=job_data.indexName, params={"keep_alive": keep_alive})["pit_id"]  # type: ignore   # noqa: E501
    try:
        reporter = get_progress_reporter(publisher)
        query["pit"] = {"id": pit_id}
        query["size"] = max_query_size

        reqs = []
        for slice_id in range(num_slices):
            written_chunks.append(os.path.join(output_dir, f"{job_data.indexName}_{slice_id}"))
            body = query.copy()
            if num_slices > 1:
                # Slice queries require max > 1
                body["slice"] = {"id": slice_id, "max": num_slices}
            res = _process_query.remote(
                {"body": body},
                search_client_args,
                job_data.fieldNames,
                written_chunks[-1],
                reporter,
                get_delimiters(),
            )
            reqs.append(res)
        results_processed = ray.get(reqs)

        if -1 in results_processed:
            raise IOError("Failed to process chunk")

        all_chunks = " ".join(written_chunks)

        annotation_path = os.path.join(output_dir, outputs.annotation)
        ret = subprocess.call(f"cat {all_chunks} > {annotation_path}; rm {all_chunks}", shell=True)
        if ret != 0:
            raise IOError(f"Failed to write {annotation_path}")

        tarball_name = os.path.basename(outputs.archived)

        ret = subprocess.call(
            f'cd {output_dir}; tar --exclude ".*" --exclude={tarball_name} -cf {tarball_name} * --remove-files',
            shell=True,
        )  # noqa: E501
        if ret != 0:
            raise IOError(f"Failed to write {outputs.archived}")
    except Exception as err:
        client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore
        raise IOError(err) from err

    client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore

    return outputs
