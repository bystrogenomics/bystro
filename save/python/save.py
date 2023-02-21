
import ray
import time
import math
import multiprocessing
from ray.util.multiprocessing import Pool
import json
import pyarrow
import gzip
from ray.cluster_utils import Cluster
import time

# Start Ray.
ray.init()

start = time.time()
def _get_slices(shards: int, max_threads: int):
    divisor = max(2, math.ceil(max_threads/shards))

    return shards * divisor

@ray.remote
def read_stuff(slice_id: int, max: int):
    import requests
    elastic_url = 'http://10.98.135.70:9200/63e38c81dd15b9001e2dbf6f_63ddc9ce1e740e0020c39928/_search?scroll=10m'
    scroll_api_url = 'http://10.98.135.70:9200/_search/scroll'
    delete_api_baseurl = 'http://10.98.135.70:9200/_search/scroll'
    headers = {'Content-Type': 'application/json'}
    print("starting ", slice_id, max)

    payload = {
        "size": 10_000,
        "sort": ["_doc"],
        "slice":{
            "id":slice_id,
            "max":max,
        },
        "query": {
            "match_all" : {}
        }
    }

    def on_error(scroll_id):
        return f"http://10.98.135.70:9200/_search/scroll/{scroll_id}"

    def initial_search_fn(payload):

        i = 0

        r1 = requests.request(
            "POST",
            elastic_url,
            data=json.dumps(payload),
            headers=headers
        )

        try:
            res_json = r1.json()
            data = res_json['hits']['hits']
            _scroll_id = res_json['_scroll_id']
            
            payload2 = {
                "scroll_id" : _scroll_id,
                "scroll": "10m",
            }
        except KeyError:
            data = []
            _scroll_id = None
            payload2 = {}

        def scroll_fn(scroll_payload):
            nonlocal i

            r2 = requests.request("GET",
                scroll_api_url,
                data=json.dumps(scroll_payload),
                headers=headers
            )

            res_json = r2.json()
            # read error as well
            if res_json.get('error'):
                print('res_json', res_json['error'])
                on_error(scroll_payload['scroll_id'])
                return []

            i += 1
            return res_json
        
        return data, _scroll_id, scroll_fn, payload2
    
    entries, scoll_id, scroll_fn, scroll_payload = initial_search_fn(payload)

    if len(entries) == 0:
        return 0

    total = 0
    last_seen = 0
    while len(entries) > 0:
        total += len(entries)
        entries = scroll_fn(scroll_payload)

        if last_seen >= 100_000:
            print(f"Have now seen: {total}")
            last_seen = 0
        last_seen += len(entries)

    r1 = requests.request(
            "DELETE",
            delete_api_baseurl,
            data=json.dumps({"scroll_id": scroll_id}),
            headers=headers
        )

    print("r1.json()", r1.json())

    return total

total = sum(ray.get([read_stuff.remote(x, 8) for x in range(8)]))
end = time.time()
print('total', total)
print("took", (end-start))