from opensearchpy import OpenSearch
from opensearchpy.helpers import scan
import ray
import time
import math
import multiprocessing
from ray.util.multiprocessing import Pool
import requests
import requests
import json
import pyarrow
import gzip
from ray.cluster_utils import Cluster


# Start Ray.
ray.init(address='auto')

def _get_slices(shards: int, max_threads: int):
    divisor = max(2, math.ceil(threads/shards))

    return shards * divisor

@ray.remote
def f(x):
    time.sleep(1)
    return x

# Start 4 tasks in parallel.
result_ids = []
for i in range(4):
    result_ids.append(f.remote(i))
    
# Wait for the tasks to complete and retrieve the results.
# With at least 4 cores, this will take 1 second.
results = ray.get(result_ids)  # [0, 1, 2, 3]

# auth = ('admin', 'admin') # For testing only. Don't store credentials in code.
# ca_certs_path = '/full/path/to/root-ca.pem' # Provide a CA bundle if you use intermediate CAs with your root CA.

# Optional client certificates if you don't want to use HTTP basic authentication.
# client_cert_path = '/full/path/to/client.pem'
# client_key_path = '/full/path/to/client-key.pem'

# Create the client with SSL/TLS enabled, but hostname verification disabled.
client = OpenSearch(
    hosts = ['http://10.98.135.70:9200'],
    http_compress = True, # enables gzip compression for request bodies
    # http_auth = auth,
    # client_cert = client_cert_path,
    # client_key = client_key_path,
    use_ssl = False,
    scheme = 'http',
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = True,
    # ca_certs = ca_certs_path
)

@ray.remote
def read_stuff(slice_id: int, max: int):
    elastic_url = 'http://10.98.135.70:9200/63e38c81dd15b9001e2dbf6f_63ddc9ce1e740e0020c39928/_search?scroll=1m'
    scroll_api_url = 'http://10.98.135.70:9200/_search/scroll'
    headers = {'Content-Type': 'application/json'}

    # client = OpenSearch(
    #     hosts = ['http://10.98.135.70:9200'],
    #     http_compress = True, # enables gzip compression for request bodies
    #     # http_auth = auth,
    #     # client_cert = client_cert_path,
    #     # client_key = client_key_path,
    #     use_ssl = False,
    #     scheme = 'http',
    #     verify_certs = False,
    #     ssl_assert_hostname = False,
    #     ssl_show_warn = True,
    #     # ca_certs = ca_certs_path
    # )

    payload = {
        "size": 100,
        "sort": ["_doc"],
        "slice":{
            "id":slice_id,
            "max":max,
        },
        "query": {
            "match_all" : {}
        }
    }

    def initial_search_fn(payload):
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
                "scroll_id" : _scroll_id
            }
        except KeyError:
            data = []
            _scroll_id = None
            payload2 = {}

        def scroll_fn():
            r2 = requests.request(
            "GET",
                scroll_api_url,
                data=json.dumps(payload2),
                headers=headers
            )

            res_json = r1.json()
            return res_json['hits']['hits']
        
        return data, _scroll_id, scroll_fn
    
    entries, scoll_id, scroll_fn = initial_search_fn(payload)

    if len(entries) == 0:
        return 0

    total = 0

    while len(entries) > 0:
        total += len(entries)
        entries = scroll_fn()

    return total

total = sum([ray.get(read_stuff.remote(x, 2)) for x in range(8)])
print('total', total)
# scroll_id = result['_scroll_id']
# total = result["hits"]["total"]

# print('total = %d' % total)

# scroll_size = len(result["hits"]["hits"])  # this is the current 'page' size
# counter = 0

# while(scroll_size > 0):
#     counter += scroll_size

    
#     scroll_id = 0
#     scroll_size = len(result['hits']['hits'])

# print('counter = %d' % counter)
# assert counter == total

# client.scroll
# Create an index with non-default settings.
# index_name = '63e38c81dd15b9001e2dbf6f_63ddc9ce1e740e0020c39928'
# index_body = {
#   'settings': {
#     'index': {
#       'number_of_shards': 4
#     }
#   }
# }
# result = es.scroll(scroll_id=0, scroll="1s")

# Initialize the scroll
# page = scan(
#     client = client,
#     index = '63e38c81dd15b9001e2dbf6f_63ddc9ce1e740e0020c39928',
#     scroll = '2m',
#     size = 1000,
#     body={"slice":{
#         "id":0,
#         "max":2,
#     }, "query": {"match_all": {}}}
# )

# page2 = scan(
#     client = client,
#     index = '63e38c81dd15b9001e2dbf6f_63ddc9ce1e740e0020c39928',
#     scroll = '2m',
#     size = 1000,
#     body={"slice":{
#         "id":1,
#         "max":2,
#     }, "query": {"match_all": {}}}
# )
# print("page", page)
# # sid = page['_scroll_id']
# scroll_size = page['hits']['total']
# print('scroll_size', scroll_size)
# print("page2 scroll_size", page2['hits']['total'])
# # print('scroll_size', scroll_size)
# # Start scrolling
# for entry in page:
#     print(entry)
# while (scroll_size['value'] > 0):
#     print("Scrolling...", scroll_size)
#     page = page.scroll(scroll_id = sid, scroll = '2m')
#     # Update the scroll ID
#     sid = page['_scroll_id']
#     # Get the number of results that we returned in the last scroll
#     scroll_size = len(page['hits']['hits'])
#     print("scroll size: " + str(scroll_size))
#     # Do something with the obtained page
# response = client.indices.create(index_name, body=index_body)
# print('\nCreating index:')
# print(response)

# Add a document to the index.
# document = {
#   'title': 'Moneyball',
#   'director': 'Bennett Miller',
#   'year': '2011'
# }
# id = '1'

# response = client.index(
#     index = index_name,
#     body = document,
#     id = id,
#     refresh = True
# )

# print('\nAdding document:')
# print(response)

# Search for the document.
# query = {
#   'size': 5,
#   'query': {
#     'match': {}
#   }
# }

# response = client.search(
#     body = query,
#     index = index_name
# )
# print('\nSearch results:')
# print(response)

# # Delete the document.
# response = client.delete(
#     index = index_name,
#     id = id
# )

# print('\nDeleting document:')
# print(response)

# # Delete the index.
# response = client.indices.delete(
#     index = index_name
# )

# print('\nDeleting index:')
# print(response)
