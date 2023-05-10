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

import mgzip # type: ignore
import numpy as np
from opensearchpy import OpenSearch
import ray

from search.utils.beanstalkd import Publisher
from search.index.handler import ProgressReporter

ray.init(ignore_reinit_error='true', address='auto')

default_delimiters = {
    'pos': "|",
    'value': ";",
    'overlap': "\\",
    'miss': "!",
    'fieldSep': "\t",
}

def make_output_names(output_base_path: str, statistics: bool,
                      compress: bool, archive: bool) -> dict:
    """Make a dictionary of output file names based on the output base path and the output options"""
    out = {}

    basename = os.path.basename(output_base_path)
    out['log'] = f"{basename}.log"
    out['annotation'] = f"{basename}.annotation.tsv"
    if compress:
        out['annotation'] += '.gz'
    out['sampleList'] = f"{basename}.sample_list"

    if statistics:
        out['statistics'] = {
            'json': f"{basename}.statistics.json",
            'tab': f"{basename}.statistics.tab",
            'qc': f"{basename}.statistics.qc.tab"
        }

    if archive:
        out['archived'] = f"{basename}.tar"

    return out


def _clamp(n, min_num, max_num):
    if n < min_num:
        return min_num

    if n > max_num:
        return max_num

    return n

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


def _make_output_string(rows, delims):
    empty_field_char = delims['miss']
    for row_idx, row in enumerate(rows): # pylint:disable=too-many-nested-blocks
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
                            inner_values.append(delims['value'].join(
                                map(lambda x: str(x) if x is not None else empty_field_char, sub)))
                        else:
                            inner_values.append(str(sub))

                    column[j] = delims['pos'].join(inner_values)

            row[i] = delims['overlap'].join(column)

        rows[row_idx] = delims['fieldSep'].join(row)

    return "\n".join(rows) + "\n"


@ray.remote
def _process_query_chunk(query_args: dict, search_client_args: dict, field_names: list,
                         chunk_output_name: str, reporter):
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
        # TODO: Complete this code for filtering
        # if filterFunctions:
        #     for f in filterFunctions:
        #         if f(doc['_source']):
        #             skipped += 1
        #             break

        row = np.empty(len(field_names), dtype=object)
        for y in range(len(field_names)):
            row[y] = _populate_data(
                child_fields[y], doc["_source"][parent_fields[y]])

        if row[discordant_idx][0][0] is False:
            row[discordant_idx][0][0] = 0
        elif row[discordant_idx][0][0] is True:
            row[discordant_idx][0][0] = 1

        rows.append(row)

    try:
        with mgzip.open(chunk_output_name, "wt", thread=8, blocksize=2*10**8) as fw:
            fw.write(_make_output_string(rows, default_delimiters))
        reporter.increment.remote(resp["hits"]["total"]["value"])
    except Exception:
        traceback.print_exc()
        return -1

    return resp["hits"]["total"]["value"]

async def go( # pylint:disable=invalid-name
    input_body: dict,
    search_conf: dict,
    publisher: Publisher,
    max_query_size: int = 10_000,
    max_slices=1024,
    keep_alive="1d",
):
    """ Main function for running the query and writing the output """
    if not input_body['compress']:
        print("\nWarning: still compressing outputs\n")

    search_client_args = dict(
        hosts=list(search_conf["connection"]["nodes"]),
        http_compress=True,
        http_auth=search_conf["auth"].get("auth"),
        client_cert=search_conf["auth"].get("client_cert_path"),
        client_key=search_conf["auth"].get("client_key_path"),
        ca_certs=search_conf["auth"].get("ca_certs_path"),
        verify_certs=True,
        ssl_assert_hostname=True,
        ssl_show_warn=True,
    )

    client = OpenSearch(**search_client_args)
    response = client.create_point_in_time( # type: ignore
        index=input_body["indexName"], params={"keep_alive": keep_alive}
    )

    pit_id = response["pit_id"]

    output_names = make_output_names(
        input_body['outputBasePath'], input_body['run_statistics'], True, input_body['archive'])

    output_dir = os.path.dirname(input_body['outputBasePath'])
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        query = _clean_query(input_body["queryBody"])
        query["pit"] = {"id": pit_id}

        query_no_sort = query.copy()

        if "sort" in query_no_sort:
            del query_no_sort["sort"]
        if "track_total_hits" in query_no_sort:
            del query_no_sort["track_total_hits"]
        del query_no_sort["pit"]
        response = client.count(
            body=query_no_sort, index=input_body["indexName"])

        n_docs = response["count"]
        assert n_docs > 0

        # minimum 2 required for this to be a slice query
        num_slices = _clamp(math.ceil(n_docs / max_query_size), 1, max_slices)

        written_chunks = [os.path.join(output_dir, f"{input_body['indexName']}_header")]

        header_output = "\t".join(input_body['fieldNames']) + "\n"
        with mgzip.open(written_chunks[-1], "wt", thread=8, blocksize=2*10**8) as fw:
            fw.write(header_output)

        query["size"] = max_query_size
        reporter = ProgressReporter.remote(publisher) # type: ignore # pylint:disable=no-member
        if num_slices == 1:
            written_chunks.append(os.path.join(
                output_dir, f"{input_body['indexName']}_{0}.gz"))
            results_processed = ray.get(
                [_process_query_chunk.remote(
                    {"body": query}, search_client_args,
                    input_body["fieldNames"], written_chunks[-1], reporter)]
            )
        else:
            save_requests = []
            for slice_id in range(num_slices):
                written_chunks.append(os.path.join(
                    output_dir, f"{input_body['indexName']}_{slice_id}"))
                query_submit = query.copy()

                query_submit["slice"] = {"id": slice_id, "max": num_slices}
                query_args = dict(
                    body=query_submit,
                    # To use sliced scrolling instead, uncomment this and remove 'pit' from query
                    # index=input_body["indexName"],
                    # params={
                    #     "scroll": "10m",
                    # }
                )
                save_requests.append(
                    _process_query_chunk.remote(
                        query_args, search_client_args,
                        input_body["fieldNames"], written_chunks[-1], reporter)
                )
            results_processed = ray.get(save_requests)

        if -1 in results_processed:
            raise IOError("Failed to process chunk")

        all_chunks = " ".join(written_chunks)

        annotation_path = os.path.join(output_dir, output_names['annotation'])
        cmd = 'cat ' + all_chunks + f'> {annotation_path}; rm {all_chunks}'
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
            raise IOError(f"Failed to write {annotation_path}")

        if input_body['archive']:
            tarball_path = os.path.join(output_dir, output_names['archived'])
            tarball_name = output_names['archived']

            cmd = f'cd {output_dir}; tar --exclude ".*" --exclude={tarball_name} -cf {tarball_name} * --remove-files'

            ret = subprocess.call(cmd, shell=True)
            if ret != 0:
                raise IOError(f"Failed to write {tarball_path}")
    except Exception as err:
        response = client.delete_point_in_time(body={"pit_id": pit_id}) # type: ignore
        raise IOError(err) from err

    return output_names
