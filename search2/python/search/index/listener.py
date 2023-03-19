import argparse
import glob
import os
import traceback
import time
from typing import Tuple

from opensearchpy.exceptions import NotFoundError
from orjson import loads, dumps
from pystalk import BeanstalkClient, BeanstalkError
from ruamel.yaml import YAML

from search.index.handler import go

# TODO: Allow passing directory for logs


def _get_config_file_path(config_path_base_dir: str, assembly, suffix: str):
    paths = glob.glob(os.path.join(config_path_base_dir,
                      assembly + suffix))

    if not paths:
        raise Exception(
            f"\n\nNo config path found for the assembly {assembly}. Exiting\n\n"
        )

    if len(paths) > 1:
        print("\n\nMore than 1 config path found, choosing first")

    return paths[0]


def _coerce_inputs(
    job_details,
    job_id,
    publisher_host,
    publisher_port,
    progress_event,
    event_queue,
    config_path_base_dir,
    search_config
) -> Tuple[dict, str]:
    mapping_config_path = _get_config_file_path(
        config_path_base_dir, job_details["assembly"], '.mapping.y*ml')
    assert len(mapping_config_path) == 1
    mapping_config_path = mapping_config_path[0]

    annotation_config_path = _get_config_file_path(
        config_path_base_dir, job_details["assembly"], '.y*ml')[0]
    assert len(annotation_config_path) == 1
    annotation_config_path = annotation_config_path[0]

    with open(mapping_config_path, 'r', encoding='utf-8') as f:
        mapping_config = YAML(typ="safe").load(f)

    with open(annotation_config_path, 'r', encoding='utf-8') as f:
        annotation_config = YAML(typ="safe").load(f)

    log_path = f"{job_details['indexName']}.index.log"

    tar_path = None
    annotation_path = None

    if job_details.get('archived'):
        tar_path = os.path.join(
            job_details['inputDir'], job_details['archived'])
    else:
        annotation_path = os.path.join(
            job_details['inputDir'], job_details['annotation'])

    return {
        "index_name": job_details["indexName"],
        "tar_path": tar_path,
        "annotation_path": annotation_path,
        "annotation_conf": annotation_config,
        "mapping_conf": mapping_config,
        "search_conf": search_config,
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
    }, log_path


def listen(queue_conf: dict, search_conf: dict, config_path_base_dir: str):
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

    tube_conf = queue_conf["tubes"]["index"]
    events_conf = queue_conf["events"]
    clients = tuple(
        (h, ports[i], BeanstalkClient(h, ports[i], socket_timeout=10))
        for i, h in enumerate(hosts)
    )

    i = 0
    while True:
        i += 1
        offset = i % len(hosts)
        host = clients[offset][0]
        port = clients[offset][1]
        client = clients[offset][2]

        client.watch(tube_conf["submission"])

        try:
            job = client.reserve_job(5)
            job_data: dict = loads(job.job_data)

            hanlder_args, log_path = _coerce_inputs(
                job_data,
                job.job_id,
                publisher_host=host,
                publisher_port=port,
                progress_event=events_conf["progress"],
                event_queue=tube_conf["events"],
                config_path_base_dir=config_path_base_dir,
                search_conf=search_conf
            )
        except BeanstalkError as err:
            if err.message == 'TIMED_OUT':
                continue
            raise err
        try:
            msg = {
                "event": events_conf["started"],
                "queueID": job.job_id,
                "submissionID": job_data["submissionID"]
            }

            client.put_job_into(tube_conf["events"], dumps(msg))
            field_names = go(**hanlder_args)

            msg['event'] = events_conf["completed"]
            msg['indexConfig'] = hanlder_args['search_config']
            msg['fieldNames'] = field_names

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
