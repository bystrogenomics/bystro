import ray
import math
import cloudpickle as pickle
import pprint
import ray
from opensearchpy import OpenSearch
import numpy as np
import traceback
import mgzip
import subprocess
import os
import pathlib

default_delimiters = {
    'pos': "|",
    'value': ";",
    'overlap': "\\",
    'miss': "!",
    'fieldSep': "\t",
}

def _clamp(n, min_num, max_num):
    if n < min_num:
        return min_num
    elif n > max_num:
        return max_num
    else:
        return n


def _get_slices(shards: int, max_threads: int):
    divisor = max(2, math.ceil(max_threads / shards))

    return shards * divisor


def _clean_query(input_query_body: dict, default_size: int = 1_000):
    # TODO: Support sort
    input_query_body["sort"] = ["_doc"]

    if "aggs" in input_query_body:
        del input_query_body["aggs"]

    if "slice" in input_query_body:
        del input_query_body["slice"]

    if "size" in input_query_body:
        del input_query_body["size"]

    print("input_query_body", input_query_body)
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
    for row_idx, row in enumerate(rows):
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
                            inner_values.append(delims['value'].join(map(lambda x: str(x) if x is not None else empty_field_char, sub)))
                        else:
                            inner_values.append(str(sub))

                    column[j] = delims['pos'].join(inner_values)

            row[i] = delims['overlap'].join(column)

        rows[row_idx] = delims['fieldSep'].join(row)

    return "\n".join(rows) + "\n"
    # if delims is None:
    #     delims = default_delimiters
    # empty_field_char = delims["miss"]
    # ouput_rows = []

    # for row in rows:
    #     output_columns = []
    #     for column in row:
    #         if column is None:
    #             column = empty_field_char
    #         else:
    #             if not isinstance(column, list):
    #                 column = [column]
    #             for position_data in column[0]:
    #                 if position_data is None:
    #                     position_data = empty_field_char
    #                 elif isinstance(position_data, list):
    #                     position_data = delims["value"].join(
    #                         [
    #                             delims["overlap"].join(sublist)
    #                             if isinstance(sublist, list)
    #                             else sublist or empty_field_char
    #                             for sublist in position_data
    #                         ]
    #                     )
    #             # column = delims["pos"].join(column[0])
    #         output_columns.append(column)
    #     ouput_rows.append(delims["fieldSep"].join(output_columns))

    # return "\n".join(ouput_rows)


@ray.remote
def _process_query_chunk(query_args: dict, search_client_args: dict, field_names: list, chunk_output_name: str):
    client = OpenSearch(**search_client_args)
    resp = client.search(**query_args)

    if resp["hits"]["total"]["value"] == 0:
        return 0
    print(resp["hits"]["total"])
    # TODO: handle 1) the munging of the data, 2) distributed pipeline/transform , 3) write as arrow table (arrow ipc format), csv

    rows = []
    # skipped = 0

    parent_fields, child_fields = _get_header(field_names)

    discordant_idx = field_names.index("discordant")
    assert discordant_idx > -1

    # Each sliced scroll chunk should get all records for that chunk
    assert len(resp["hits"]["hits"]) == resp["hits"]["total"]["value"]

    for doc in resp["hits"]["hits"]:
        # TODO: implement distributed filters
        # if filterFunctions:
        #     for f in filterFunctions:
        #         if f(doc['_source']):
        #             skipped += 1
        #             break

        row = np.empty(len(field_names), dtype=object)
        for y in range(len(field_names)):
            row[y] = _populate_data(child_fields[y], doc["_source"][parent_fields[y]])

        if row[discordant_idx][0][0] == False:
            row[discordant_idx][0][0] = 0
        elif row[discordant_idx][0][0] == True:
            row[discordant_idx][0][0] = 1

        rows.append(row)

    try:
        with mgzip.open(chunk_output_name, "wt", thread=8, blocksize=2*10**8) as fw:
            fw.write(_make_output_string(rows, default_delimiters))
    except Exception:
        traceback.print_exc()

    return resp["hits"]["total"]["value"]


# TODO: get max_slices from opensearch index settings
def go(
    input_body: dict,
    search_conf: dict,
    max_query_size: int = 10_000,
    max_slices=1024,
    keep_alive="1d",
):
    pp = pprint.PrettyPrinter(indent=4)
    print("\n\ngot input")
    pp.pprint(input_body)

    search_client_args = dict(
        hosts=list(search_conf["connection"]["nodes"]),
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=search_conf["auth"].get("auth"),
        client_cert=search_conf["auth"].get("client_cert_path"),
        client_key=search_conf["auth"].get("client_key_path"),
        ca_certs=search_conf["auth"].get("ca_certs_path"),
        verify_certs=True,
        ssl_assert_hostname=True,
        ssl_show_warn=True,
    )

    client = OpenSearch(**search_client_args)
    response = client.create_point_in_time(
        index=input_body["indexName"], params={"keep_alive": keep_alive}
    )

    pit_id = response["pit_id"]

    annotation_path = f"{input_body['outputBasePath']}.annotation.tsv"
    output_dir = os.path.dirname(annotation_path)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if input_body['compress']:
        annotation_path += '.gz'

    try:
        query = _clean_query(input_body["queryBody"])
        query["pit"] = {"id": pit_id}

        query_no_sort = query.copy()

        if "sort" in query_no_sort:
            del query_no_sort["sort"]
        del query_no_sort["pit"]
        response = client.count(body=query_no_sort, index=input_body["indexName"])

        n_docs = response["count"]
        assert n_docs > 0

        # minimum 2 required for this to be a slice query
        num_slices = _clamp(math.ceil(n_docs / max_query_size), 2, max_slices)

        # TODO: concatenate chunks in a different ray worker
        written_chunks = [f"{input_body['indexName']}_header"]

        header_output = "\t".join(input_body['fieldNames']) + "\n"
        with mgzip.open(written_chunks[-1], "wt", thread=8, blocksize=2*10**8) as fw:
            fw.write(header_output)

        query["size"] = max_query_size
        if num_slices == 1:
            written_chunks.append(f"{input_body['indexName']}_{0}.gz")
            results_processed = ray.get(
                [_process_query_chunk.remote({"body": query}, search_client_args, input_body["fieldNames"], written_chunks[-1])]
            )
        else:
            save_requests = []
            for slice_id in range(num_slices):
                written_chunks.append(f"{input_body['indexName']}_{slice_id}")
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
                    _process_query_chunk.remote(query_args, search_client_args, input_body["fieldNames"], written_chunks[-1])
                )
            results_processed = ray.get(save_requests)

        total = sum(results_processed)

        cmd = 'cat ' + " ".join(written_chunks) + f'> {annotation_path}'
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
            raise Exception(f"Failed to write outputs for {input_body['indexName']}")
        print("total", total)
    except Exception as err:
        response = client.delete_point_in_time(body={"pit_id": pit_id})
        print("delete response", response)
        raise Exception(err) from err
