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
import pickle
import json
from collections.abc import Iterable
from pystalk import BeanstalkClient, BeanstalkError
from handler import go
from typing import ByteString
from os import path
import pprint
from orjson import loads, dumps
from ruamel.yaml import YAML
import argparse
import glob
from opensearchpy.exceptions import NotFoundError
import traceback
import time
required_keys = ("outputBasePath", "assembly", "queryBody", "fieldNames", "indexName")
optional_keys = ("indexConfig", "pipeline")


def _coerce_inputs(
    job_details,
    job_id,
    publisher_host,
    publisher_port,
    progress_event,
    event_queue,
    config_path_base_dir,
):
    job_specific_args = {}

    for key in required_keys:
        if key not in job_details:
            raise Exception(f"Missing required key: {key} in job message")

        job_specific_args[key] = job_details[key]

    for key in optional_keys:
        if key in job_details:
            job_specific_args[key] = job_details[key]

    config_file_path = _get_config_file_path(
        config_path_base_dir, job_specific_args["assembly"]
    )

    if not config_file_path:
        raise Exception(
            f"Assembly {job_specific_args['assembly']} doesn't have corresponding config file"
        )

    common_args = {
        "config": config_file_path,
        "publisher": {
            "host": publisher_host,
            "port": publisher_port,
            "queue": event_queue,
            "messageBase": {
                "event": progress_event,
                "queueID": job_id,
                "submissionID": job_details["submissionID"],
                "data": None,
            },
        },
        "compress": True,
        "archive": True,
        "run_statistics": True,
    }

    combined_args = {**common_args, **job_specific_args}

    return combined_args


def _get_config_file_path(config_path_base_dir: str, assembly):
    paths = glob.glob(path.join(config_path_base_dir, assembly + ".y*ml"))

    if not paths:
        raise Exception(
            f"\n\nNo config path found for the assembly {assembly}. Exiting\n\n"
        )

    if len(paths) > 1:
        print("\n\nMore than 1 config path found, choosing first")

    return paths[0]


def listen(queue_conf: dict, search_conf: dict, config_path_base_dir: str):
    """TODO: Listen on beanstalkd here directly"""
    queue_conf = queue_conf["beanstalkd"]

    if isinstance(queue_conf["host"], str):
        hosts = (queue_conf["host"],)
        ports = (queue_conf["port"],)
    else:
        hosts = tuple(queue_conf["host"])
        ports = tuple(queue_conf["port"])

    assert isinstance(hosts, tuple) and isinstance(ports, tuple)

    for event in ("progress", "failed", "started", "completed"):
        assert event in queue_conf["events"]

    tube_conf = queue_conf["tubes"]["saveFromQuery"]
    events_conf = queue_conf["events"]
    clients = tuple(
        (h, ports[i], BeanstalkClient(h, ports[i], socket_timeout=10))
        for i, h in enumerate(hosts)
    )

    pp = pprint.PrettyPrinter(indent=4)

    i = 0
    while True:
        i += 1
        offset = i % len(hosts)
        host = clients[offset][0]
        port = clients[offset][1]
        client = clients[offset][2]

        client.watch(tube_conf["submission"])
        print('tube_conf["submission"]', tube_conf["submission"])
        try:
            job = client.reserve_job(5)
            job_data: dict = loads(job.job_data)

            # create the annotator
            input_data: dict = _coerce_inputs(
                job_data,
                job.job_id,
                publisher_host=host,
                publisher_port=port,
                progress_event=events_conf["progress"],
                event_queue=tube_conf["events"],
                config_path_base_dir=config_path_base_dir,
            )
        except BeanstalkError as err:
            if err.message == "TIMED_OUT":
                continue
            raise err
        try:
            with open(input_data['config'], 'r', encoding='utf-8') as f:
                job_config = YAML(typ="safe").load(f)

            msg = {
                "event": events_conf["started"],
                "jobConfig": job_config,
                "queueID": job.job_id,
                "submissionID": job_data["submissionID"]
            }

            client.put_job_into(tube_conf["events"], dumps(msg))
            output_names = go(input_data, search_conf)

            del msg['jobConfig']
            msg['results'] = {"outputFileNames": output_names}
            msg['event'] = events_conf["completed"]

            client.put_job_into(tube_conf["events"], dumps(msg))
            client.delete_job(job.job_id)
        except BeanstalkError as err:
            print(f"Received error during execution: {err}")
            client.release_job(job.job_id)
        except NotFoundError as err:
            msg = {
                "event": events_conf["failed"],
                "reason": "404",
                "queueID": job.job_id,
                "submissionID": job_data["submissionID"]
            }

            traceback.print_exc()
            client.put_job_into(tube_conf["events"], dumps(msg))
            client.delete_job(job.job_id)
        except Exception as err:
            traceback.print_exc()
            client.release_job(job.job_id)
        finally:
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Process some config files.")
    parser.add_argument(
        "--conf_dir", type=str, help="Path to the genome/assembly config directory"
    )
    parser.add_argument(
        "--queue_conf",
        type=str,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
    )
    parser.add_argument(
        "--search_conf",
        type=str,
        help="Path to the opensearch config yaml file (e.g. elasticsearch.yml)",
    )
    args = parser.parse_args()

    config_path_base_dir = args.conf_dir
    with open(args.queue_conf, "r", encoding="utf-8") as queue_config_file:
        queue_conf = YAML(typ="safe").load(queue_config_file)

    with open(args.search_conf, "r", encoding="utf-8") as search_config_file:
        search_conf = YAML(typ="safe").load(search_config_file)
    listen(queue_conf, search_conf, config_path_base_dir)


if __name__ == "__main__":
    main()
