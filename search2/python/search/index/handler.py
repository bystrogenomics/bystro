import argparse
from math import ceil
import os

from opensearchpy import OpenSearch, helpers, exceptions
from orjson import dumps
from pystalk import BeanstalkClient, BeanstalkError
from ruamel.yaml import YAML

from search.index.bystro_file import read_annotation_tarball

# TODO: allow reading directly from annotation_path or get rid of that possibility in annotator
# TODO: read delimiters from annotation_conf
class ProgressReporter:
    def __init__(self, publisher: dict):
        self.value = 0
        self.publisher = publisher.copy()
        self.client = BeanstalkClient(publisher['host'], publisher['port'], socket_timeout=10)

    def increment(self, count: int):
        self.value += count
        self.publisher['messageBase']['data'] = {
            "progress": self.value,
            "skipped": 0
        }

        try:
            self.client.put_job_into(self.publisher['queue'], dumps(self.publisher['messageBase']))
        except BeanstalkError as err:
            raise err

        return self.value

    def get_counter(self):
        return self.value

class ProgressReporterStub:
    def __init__(self):
        self.value = 0

    def increment(self, count: int):
        self.value += count
        print(f"Processed {self.value} records")

    def get_counter(self):
        return self.value

def go(
        index_name: str,
        tar_path: str,
        annotation_conf: dict,
        mapping_conf: dict,
        search_conf: dict,
        chunk_size=5000,
        publisher: dict = None,
        annotation_path: str = None
):
    assert not (tar_path is not None and annotation_path is not None)
    assert tar_path is not None

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

    if publisher:
        reporter = ProgressReporter(publisher)
    else:
        reporter = ProgressReporterStub()

    client = OpenSearch(**search_client_args)

    post_index_settings = mapping_conf['post_index_settings']
    boolean_map = {x: True for x in mapping_conf['booleanFields']}

    index_body = {
        'settings': mapping_conf['index_settings'],
        'mappings': mapping_conf['mappings'],
    }

    delimiters={
        'field': "\t",
        "allele": ",",
        "position": "|",
        "value": ";",
        "empty_field": "!"
    }

    if not index_body['settings'].get('number_of_shards'):
          file_size = os.path.getsize(tar_path)
          index_body['settings']['number_of_shards'] = ceil(float(file_size) / float(1e10))

    try:
        client.indices.create(index_name, body=index_body)
    except exceptions.RequestError as e:
        print(e)

    data = read_annotation_tarball(index_name=index_name, tar_path=tar_path,
                                   boolean_map=boolean_map, delimiters=delimiters,
                                   chunk_size=chunk_size)
    helpers.parallel_bulk(client, data)

    count = 0
    for response in helpers.parallel_bulk(client, data, chunk_size=chunk_size):
        assert response[0] is True
        count += 1
        if count >=chunk_size:
            reporter.increment(count)
            count = 0

    result = client.indices.put_settings(index=index_name, body=post_index_settings)
    print(result)

    return data.header_fields, index_body['mappings']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some config files.")
    parser.add_argument(
        "--tar", type=str, help="Path to the tarball containing the annotation"
    )
    parser.add_argument(
        "--annotation_conf", type=str, help="Path to the genome/assembly config, e.g. hg19.yml"
    )

    parser.add_argument(
        "--search_conf",
        type=str,
        help="Path to the opensearch config yaml file (e.g. elasticsearch.yml)",
    )

    parser.add_argument(
        "--mapping_conf",
        type=str,
        help="Path to the opensearch config yaml file (e.g. hg19.mapping.yml)",
    )

    parser.add_argument(
        "--index_name",
        type=str,
        help="Opensearch index name",
    )

    args = parser.parse_args()

    with open(args.annotation_conf, "r", encoding="utf-8") as f:
        annotation_conf = YAML(typ="safe").load(f)

    with open(args.search_conf, "r", encoding="utf-8") as f:
        search_conf = YAML(typ="safe").load(f)

    with open(args.mapping_conf, "r", encoding="utf-8") as f:
        mapping_conf = YAML(typ="safe").load(f)

    go(
        index_name=args.index_name,
        tar_path=args.tar,
        annotation_conf=annotation_conf,
        mapping_conf=mapping_conf,
        search_conf=search_conf)