"""TODO: Add description here"""
from glob import glob
from os import path
import time
import traceback
from typing import Any, NamedTuple, List, Callable, Optional

import asyncio
from orjson import loads, dumps  # pylint: disable=no-name-in-module
from pystalk import BeanstalkClient, BeanstalkError
from opensearchpy.exceptions import NotFoundError

import ray

BEANSTALK_ERR_TIMEOUT = "TIMED_OUT"

class Message(NamedTuple):
    """Beanstalkd Message"""
    event: str
    queueID: str
    submissionID: str
    data: Any = None

class Publisher(NamedTuple):
    """Beanstalkd Message Published Config"""
    host: str
    port: str
    queue: str
    message: Message

class QueueConf(NamedTuple):
    """Queue Configuration"""
    addresses: List[str]
    events: dict
    tubes: dict

    def split_host_port(self):
        """TODO: Add description here"""
        hosts = []
        ports = []
        for host in self.addresses:
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
    """Listen on a Beanstalkd channel, waiting for work.
       When work is available call the work handler
    """
    hosts, ports = queue_conf.split_host_port()

    for event in ("progress", "failed", "started", "completed"):
        assert event in queue_conf.events

    tube_conf = queue_conf.tubes[tube]
    events_conf = queue_conf.events
    clients = tuple((host, port, BeanstalkClient(host, port, socket_timeout=10)) for (host, port) in zip(hosts, ports))

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
            if err.message == BEANSTALK_ERR_TIMEOUT:
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

@ray.remote(num_cpus=0)
class ProgressReporter:
    """A Ray class to report progress to a beanstalk queue"""

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
    """A Ray class to report progress to stdout"""

    def __init__(self):
        self.value = 0

    def increment(self, count: int):
        self.value += count
        print(f"Processed {self.value} records")

    def get_counter(self):
        return self.value

def get_progress_reporter(publisher: Optional[Publisher]):
    if publisher:
        return ProgressReporter.remote(publisher)  # type: ignore # pylint: disable=no-member

    return ProgressReporterStub.remote()  # type: ignore # pylint: disable=no-member