"""TODO: Add description here"""
from glob import glob
from os import path
import time
import traceback
from typing import Any, NamedTuple, List, Callable, Optional

import asyncio
from orjson import loads, dumps  # pylint: disable=no-name-in-module
from pystalk import BeanstalkClient, BeanstalkError  # type: ignore
from opensearchpy.exceptions import NotFoundError

class Message(NamedTuple):
    """TODO: Add description here"""
    event: str
    queueID: str
    submissionID: str
    data: Optional[Any]

class Publisher(NamedTuple):
    """TODO: Add description here"""

    host: str
    port: str
    queue: str
    message: Message

class QueueConf(NamedTuple):
    """TODO: Add description here"""

    address: List[str]
    events: dict
    tubes: dict

    def split_host_port(self):
        """TODO: Add description here"""
        hosts = []
        ports = []
        for host in self.address:
            host, port = host.split(":")
            hosts.append(host)
            ports.append(port)
        return hosts, ports

def get_config_file_path(config_path_base_dir: str, assembly: str, suffix: str = ".y*ml"):
    """TODO: Add description here"""
    paths = glob(path.join(config_path_base_dir, assembly + suffix))

    if not paths:
        raise ValueError(
            f"\n\nNo config path found for the assembly {assembly}. Exiting\n\n"
        )

    if len(paths) > 1:
        print("\n\nMore than 1 config path found, choosing first")

    return paths[0]

def _specify_publisher(
    submission_id, job_id, publisher_host, publisher_port, progress_event, event_queue
):
    return Publisher(
            host = publisher_host,
            port = publisher_port,
            queue = event_queue,
            message = Message(
                event = progress_event,
                queueID = job_id,
                submissionID = submission_id,
                data = None,
            ))

def listen(
    handler_fn: Callable,
    submit_msg_fn: Callable,
    completed_msg_fn: Callable,
    queue_conf: QueueConf,
    tube: str,
):
    """TODO: Listen on beanstalkd here directly"""
    hosts, ports = queue_conf.split_host_port()

    for event in ("progress", "failed", "started", "completed"):
        assert event in queue_conf.events

    tube_conf = queue_conf.tubes[tube]
    events_conf = queue_conf.events
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
        except BeanstalkError as err:
            if err.message == "TIMED_OUT":
                continue
            raise err
        try:
            job_data = loads(job.job_data)

            publisher: Publisher = _specify_publisher(
                job_data['submissionID'],
                job.job_id,
                publisher_host=host,
                publisher_port=port,
                progress_event=events_conf["progress"],
                event_queue=tube_conf["events"]
            )

            base_msg = {
                "event": events_conf["started"],
                "queueID": job.job_id,
                "submissionID": job_data["submissionID"],
            }

            msg = submit_msg_fn(base_msg, job_data)

            client.put_job_into(tube_conf["events"], dumps(msg))
            res = asyncio.get_event_loop().run_until_complete(handler_fn(publisher, job_data))

            base_msg["event"] = events_conf["completed"]
            msg = completed_msg_fn(base_msg, job_data, res)

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
                "submissionID": job_data["submissionID"],
            }

            traceback.print_exc()
            client.put_job_into(tube_conf["events"], dumps(msg))
            client.delete_job(job.job_id)
        except Exception as err:
            traceback.print_exc()
            client.release_job(job.job_id)
        finally:
            time.sleep(0.5)
