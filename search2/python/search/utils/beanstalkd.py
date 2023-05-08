"""TODO: Add description here"""
import time
import traceback
from typing import Any, NamedTuple, List, Callable, Optional

from orjson import loads, dumps  # pylint: disable=no-name-in-module
from pystalk import BeanstalkClient, BeanstalkError  # type: ignore
from ruamel.yaml import YAML
from opensearchpy.exceptions import NotFoundError

class Message(NamedTuple):
    """TODO: Add description here"""
    event: str
    ququeID: str
    submissionID: str
    data: Optional[Any]

class Publisher(NamedTuple):
    """TODO: Add description here"""

    host: str
    port: str
    queue: str
    tubes: dict
    message: Message

class QueueConf(NamedTuple):
    """TODO: Add description here"""

    host: List[str]
    port: List[str]
    events: dict
    tubes: dict
    beanstalkd: dict


def _specify_publisher(
    job_details, job_id, publisher_host, publisher_port, progress_event, event_queue
):
    Publisher(
            host = publisher_host,
            port = publisher_port,
            queue = event_queue,
            message = Message(
                event = progress_event,
                queueID = job_id,
                submissionID = job_details["submissionID"],
                data = None,
            ))

def listen(
    handler_fn: Callable,
    msg_fn: Callable,
    queue_conf: QueueConf,
    search_conf: dict,
    tube: str,
):
    """TODO: Listen on beanstalkd here directly"""
    hosts = queue_conf.host
    ports = queue_conf.port

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
            job_data: dict = loads(job.job_data)

            publisher: Publisher = _specify_publisher(
                job_data,
                job.job_id,
                publisher_host=host,
                publisher_port=port,
                progress_event=events_conf["progress"],
                event_queue=tube_conf["events"]
            )
        except BeanstalkError as err:
            if err.message == "TIMED_OUT":
                continue
            raise err
        try:
            msg = {
                "event": events_conf["started"],
                "jobConfig": job_config,
                "queueID": job.job_id,
                "submissionID": job_data["submissionID"],
            }

            client.put_job_into(tube_conf["events"], dumps(msg))
            res = asyncio.get_event_loop().run_until_complete(handler_fn(publisher))
            output_names = fn(input_data, search_conf)

            del msg["jobConfig"]
            msg["results"] = {"outputFileNames": output_names}
            msg["event"] = events_conf["completed"]

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
