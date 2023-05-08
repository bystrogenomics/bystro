# TODO: allow reading directly from annotation_path or get rid of that possibility in annotator
# TODO: read delimiters from annotation_conf

import argparse
import asyncio
from math import ceil
import multiprocessing
import os
import time

from opensearchpy._async.client import AsyncOpenSearch
from opensearchpy._async import helpers as async_helpers
from orjson import dumps  # pylint: disable=no-name-in-module
from pystalk import BeanstalkClient  # type: ignore
from ruamel.yaml import YAML

from search.index.bystro_file import (
    read_annotation_tarball,
)  # pylint: disable=no-name-in-module,import-error
from search.utils.beanstalkd import Publisher

import ray

ray.init(ignore_reinit_error="true", address="auto")

n_threads = multiprocessing.cpu_count()


@ray.remote
class Indexer:
    """TODO: Add description here"""

    def __init__(
        self,
        search_client_args,
        progress_tracker,
        chunk_size=1_000,
        reporter_batch=15_000,
    ):
        self.search_client_args = search_client_args
        self.client = AsyncOpenSearch(**self.search_client_args)
        self.progress_tracker = progress_tracker
        self.reporter_batch = reporter_batch
        self.chunk_size = chunk_size
        self.counter = 0

    async def run(self, data):
        resp = await async_helpers.async_bulk(
            self.client, iter(data), chunk_size=self.chunk_size
        )
        self.counter += resp[0]

        if self.counter >= self.reporter_batch:
            await asyncio.to_thread(
                self.progress_tracker.increment.remote, self.counter
            )
            self.counter = 0

        return resp

    async def close(self):
        await self.client.close()


@ray.remote(num_cpus=0)
class ProgressReporter:
    """A class to report progress to a beanstalk queue"""

    def __init__(self, publisher: Publisher):
        self.value = 0
        self.publisher = publisher
        self.message = publisher.message._asdict()

        self.message["data"] = {"progress": 0, "skipped": 0}

        self.client = BeanstalkClient(publisher.host, publisher.port, socket_timeout=10)

    def increment(self, count: int):
        """Increment the counter by processed variant count and report to the beanstalk queue"""
        self.value += count
        self.message["data"]["progress"] = self.value

        self.client.put_job_into(self.publisher.queue, dumps(self.message))

        return self.value

    def get_counter(self):
        """Get the current value of the counter"""
        return self.value


@ray.remote(num_cpus=0)
class ProgressReporterStub:
    def __init__(self):
        self.value = 0

    def increment(self, count: int):
        self.value += count
        print(f"Processed {self.value} records")

    def get_counter(self):
        return self.value


async def go(
    index_name: str,
    tar_path: str,
    mapping_conf: dict,
    search_conf: dict,
    chunk_size=500,
    paralleleism_chunk_size=5_000,
    publisher: Publisher = None,
    annotation_path: str = None,
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
        reporter = ProgressReporter.remote(publisher)  # pylint: disable=no-member
    else:
        reporter = ProgressReporterStub.remote()  # pylint: disable=no-member

    client = AsyncOpenSearch(**search_client_args)

    post_index_settings = mapping_conf["post_index_settings"]
    boolean_map = {x: True for x in mapping_conf["booleanFields"]}

    index_body = {
        "settings": mapping_conf["index_settings"],
        "mappings": mapping_conf["mappings"],
    }

    delimiters = {
        "field": "\t",
        "allele": ",",
        "position": "|",
        "value": ";",
        "empty_field": "!",
    }

    if not index_body["settings"].get("number_of_shards"):
        file_size = os.path.getsize(tar_path)
        index_body["settings"]["number_of_shards"] = ceil(
            float(file_size) / float(1e10)
        )

    try:
        await client.indices.create(index_name, body=index_body)
    except Exception as e:
        print(e)

    data = read_annotation_tarball(
        index_name=index_name,
        tar_path=tar_path,
        boolean_map=boolean_map,
        delimiters=delimiters,
        chunk_size=paralleleism_chunk_size,
    )

    start = time.time()
    actors = [
        Indexer.remote( # pylint: disable=no-member
            search_client_args, progress_tracker=reporter, chunk_size=chunk_size
        )
        for x in range(n_threads)
    ]
    actor_idx = 0
    results = []
    for x in data:
        actor = actors[actor_idx]
        actor_idx += 1
        if actor_idx == n_threads:
            actor_idx = 0
        results.append(actor.run.remote(x))
    res = ray.get(results)

    print("Took", time.time() - start)

    errors = []
    total = 0
    for x in res: # pylint: disable=invalid-name
        total += x[0]
        if x[1]:
            errors.append(",".join(x[1]))

    if errors:
        raise ValueError("\n".join(errors))

    print(f"Processed {total} records")

    for actor in actors:
        await actor.close.remote()

    result = await client.indices.put_settings(
        index=index_name, body=post_index_settings
    )

    print(result)

    await client.close()

    return data.get_header_fields()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some config files.")
    parser.add_argument(
        "--tar", type=str, help="Path to the tarball containing the annotation"
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

    with open(args.search_conf, "r", encoding="utf-8") as f:
        search_conf = YAML(typ="safe").load(f)

    with open(args.mapping_conf, "r", encoding="utf-8") as f:
        mapping_conf = YAML(typ="safe").load(f)

    asyncio.run(
        go(
            index_name=args.index_name,
            tar_path=args.tar,
            mapping_conf=mapping_conf,
            search_conf=search_conf,
        )
    )
