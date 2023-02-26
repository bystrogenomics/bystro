import ray
import math
import cloudpickle as pickle
import pprint
import ray
from opensearchpy import OpenSearch

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

    if not "size" in input_query_body:
        input_query_body["size"] = default_size

    print("input_query_body", input_query_body)
    return input_query_body


@ray.remote
def _process_query_chunk(query_args: dict, search_client_args: dict):
    client = OpenSearch(**search_client_args)

    resp = client.search(**query_args)
    print(resp['hits']['total'])
    # TODO: handle 1) the munging of the data, 2) distributed pipeline/transform , 3) write as arrow table (arrow ipc format), csv
    # variant_records = resp['hits']['hits']

    return resp['hits']['total']['value']


def go(input_body: dict, search_conf: dict, max_slices: int = 2, keep_alive="1d"):
    pp = pprint.PrettyPrinter(indent=4)
    print("\n\ngot input")
    pp.pprint(input_body)
    print("\n\ngot search_conf")
    pp.pprint(search_conf)

    # TODO: get from number of records, shards
    num_slices = max_slices

    search_client_args = dict(
        hosts = list(search_conf["connection"]["nodes"]),
        http_compress=True,  # enables gzip compression for request bodies
        http_auth = search_conf["auth"].get('auth'),
        client_cert = search_conf["auth"].get('client_cert_path'),
        client_key = search_conf["auth"].get('client_key_path'),
        ca_certs = search_conf["auth"].get('ca_certs_path'),
        verify_certs = True,
        ssl_assert_hostname = True,
        ssl_show_warn = True
    )

    client = OpenSearch(**search_client_args)
    response = client.create_point_in_time(index=input_body['indexName'], params= dict(keep_alive=keep_alive))

    pit_id = response["pit_id"]

    try:
        query = _clean_query(input_body["queryBody"])
        query['pit'] = {
            "id": pit_id
        }

        save_requests = []
        for slice_id in range(num_slices):
            query_submit = query.copy()
            query_submit['slice'] = {
                "id": slice_id,
                "max": num_slices
            }

            query_args = dict(
                body=query_submit,
                # To use sliced scrolling instead, uncomment this and remove 'pit' from query
                # index=input_body["indexName"],
                # params={
                #     "scroll": "10m",
                # }
            )

            save_requests.append(_process_query_chunk.remote(query_args, search_client_args))
        results_processed = ray.get(save_requests)

        total = sum(results_processed)
        print("total", total)
    except Exception as err:
        response = client.delete_point_in_time(body={
            "pit_id": pit_id
        })
        print(response)
        raise Exception(err) from err
