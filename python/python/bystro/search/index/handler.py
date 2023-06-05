# TODO: read delimiters from annotation_conf

import argparse
import asyncio
import multiprocessing
import os
import time
from math import ceil

import ray
from opensearchpy._async import helpers as async_helpers
from opensearchpy._async.client import AsyncOpenSearch
from ruamel.yaml import YAML

from bystro.search.index.bystro_file import read_annotation_tarball  # type: ignore # pylint: disable=no-name-in-module,import-error  # noqa: E501
from bystro.search.utils.annotation import get_delimiters
from bystro.search.utils.beanstalkd import ProgressPublisher, get_progress_reporter
from bystro.search.utils.opensearch import gather_opensearch_args

ray.init(ignore_reinit_error=True, address="auto")

n_threads = multiprocessing.cpu_count()


@ray.remote
class Indexer:
    """Ray Actor to handle indexing of Bystro annotation data into OpenSearch"""

    def __init__(
        self,
        search_client_args,
        progress_tracker,
        chunk_size=1_000,
        reporter_batch=15_000,
    ):
        self.client = AsyncOpenSearch(**search_client_args)
        self.progress_tracker = progress_tracker
        self.reporter_batch = reporter_batch
        self.chunk_size = chunk_size
        self.counter = 0

    async def index(self, data):
        """Index Bystro annotation data into Opensearch"""
        resp = await async_helpers.async_bulk(self.client, iter(data), chunk_size=self.chunk_size)
        self.counter += resp[0]

        if self.counter >= self.reporter_batch:
            await asyncio.to_thread(self.progress_tracker.increment.remote, self.counter)
            self.counter = 0

        return resp

    async def close(self):
        await self.client.close()


async def go(
    index_name: str,
    mapping_conf: dict,
    search_conf: dict,
    tar_path: str,
    chunk_size=500,
    paralleleism_chunk_size=5_000,
    publisher: ProgressPublisher | None = None,
) -> list[str]:
    """Main handler for indexing Bystro annotation data into OpenSearch

    Args:
        index_name (str): _description_
        mapping_conf (dict): _description_
        search_conf (dict): _description_
        tar_path (str): _description_
        chunk_size (int, optional): _description_. Defaults to 500.
        paralleleism_chunk_size (_type_, optional): _description_. Defaults to 5_000.
        publisher (Optional[Publisher], optional): _description_. Defaults to None.

    Raises:
        RuntimeError: _description_

    Returns:
        _type_: _description_
    """
    reporter = get_progress_reporter(publisher)

    search_client_args = gather_opensearch_args(search_conf)
    client = AsyncOpenSearch(**search_client_args)

    post_index_settings = mapping_conf["post_index_settings"]
    boolean_map: dict[str, bool] = {x: True for x in mapping_conf["booleanFields"]}

    index_body: dict[str, dict] = {
        "settings": mapping_conf["index_settings"],
        "mappings": mapping_conf["mappings"],
    }

    if not index_body["settings"].get("number_of_shards"):
        file_size = os.path.getsize(tar_path)
        index_body["settings"]["number_of_shards"] = ceil(float(file_size) / float(1e10))

    try:
        await client.indices.create(index_name, body=index_body)
    except Exception as e:
        print(e)

    data = read_annotation_tarball(
        index_name=index_name,
        tar_path=tar_path,
        boolean_map=boolean_map,
        delimiters=get_delimiters(),
        chunk_size=paralleleism_chunk_size,
    )

    start = time.time()
    indexers = [Indexer.remote(search_client_args, reporter, chunk_size) for _ in range(n_threads)]  # type: ignore  # noqa: E501
    actor_idx = 0
    results = []
    for x in data:
        "Round robin work distribution"
        indexer = indexers[actor_idx]
        actor_idx += 1
        if actor_idx == n_threads:
            actor_idx = 0
        results.append(indexer.index.remote(x))
    res = ray.get(results)

    reported_count: int = ray.get(reporter.get_counter.remote())  # type: ignore

    errors = []
    total = 0
    for indexed_count, errors in res:
        total += indexed_count
        if errors:
            errors.append(",".join(errors))

    if errors:
        raise RuntimeError("\n".join(errors))

    to_report_count = total - reported_count
    if to_report_count > 0:
        await asyncio.to_thread(reporter.increment.remote, to_report_count)  # type: ignore

    print(f"Processed {total} records in {time.time() - start}s")

    for indexer in indexers:
        await indexer.close.remote()

    await client.indices.put_settings(index=index_name, body=post_index_settings)
    await client.close()

    return data.get_header_fields()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some config files.")
    parser.add_argument("--tar", type=str, help="Path to the tarball containing the annotation")

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
        search_config = YAML(typ="safe").load(f)

    with open(args.mapping_conf, "r", encoding="utf-8") as f:
        mapping_config = YAML(typ="safe").load(f)

    asyncio.run(go(args.index_name, mapping_config, search_config, tar_path=args.tar))
