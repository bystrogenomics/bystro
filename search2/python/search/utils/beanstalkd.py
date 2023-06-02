"""TODO: Add description here"""
import abc
from copy import deepcopy
from enum import Enum
from glob import glob
from os import path
import time
import traceback
from typing import Any, NamedTuple, List, Callable, Union

import asyncio
from msgspec import Struct, json
from pystalk import BeanstalkClient, BeanstalkError
from opensearchpy.exceptions import NotFoundError

import ray
from ray.types import ObjectRef

BEANSTALK_ERR_TIMEOUT = "TIMED_OUT"

class BaseMessage(Struct):
    submissionID: str

class FailedMessage(BaseMessage):
    reason: str

class Event(str, Enum):
    """Beanstalkd Event"""
    PROGRESS = "progress"
    FAILED = "failed"
    STARTED = "started"
    COMPLETED = "completed"

class ProgressData(Struct):
    progress: int = 0
    skipped: int = 0

class ProgressMessage(BaseMessage):
    """Beanstalkd Message"""
    event: str = Event.PROGRESS
    data: ProgressData | str | None = None

class ProgressPublisher(NamedTuple):
    """Beanstalkd Message Published Config"""
    host: str
    port: int
    queue: str
    message: ProgressMessage

class QueueConf(Struct):
    """Queue Configuration"""
    addresses: List[str]
    events: Event
    tubes: dict

    def split_host_port(self):
        """Split host and port"""
        hosts = []
        ports = []
        for host in self.addresses:
            host, port = host.split(":")
            hosts.append(host)
            ports.append(port)
        return hosts, ports

def get_config_file_path(config_path_base_dir: str, assembly: str, suffix: str = ".y*ml"):
    """Get config file path"""
    paths = glob(path.join(config_path_base_dir, assembly + suffix))

    if not paths:
        raise ValueError(f"\n\nNo config path found for the assembly {assembly}. Exiting\n\n")

    if len(paths) > 1:
        print("\n\nMore than 1 config path found, choosing first")

    return paths[0]

def listen(
    job_data_type: BaseMessage,
    handler_fn: Callable[[ProgressPublisher, BaseMessage], Any],
    submit_msg_fn: Callable[[BaseMessage], BaseMessage],
    completed_msg_fn: Callable[[BaseMessage, Any], BaseMessage],
    failed_msg_fn: Callable[[BaseMessage, Exception], FailedMessage],
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
    clients = tuple(BeanstalkClient(host, port, socket_timeout=10) for (host, port) in zip(hosts, ports))

    i = 0
    while True:
        i += 1
        offset = i % len(hosts)
        client  = clients[offset]

        client.watch(tube_conf["submission"])
        client.use(tube_conf["events"])

        try:
            job = client.reserve_job(5)
        except BeanstalkError as err:
            if err.message == BEANSTALK_ERR_TIMEOUT:
                continue
            raise err

        try:
            job_data: BaseMessage = json.decode(job.body, type=job_data_type)
        except Exception as err:
            print(f"Received invalid JSON: {err}")
            client.delete_job(job.job_id)
            continue

        try:
            publisher = ProgressPublisher(
                        host = client.host,
                        port = client.port,
                        queue = tube_conf["events"],
                        message = ProgressMessage(
                            submissionID = job_data.submissionID
                        ))

            msg = submit_msg_fn(job_data)

            client.put_job(json.encode(msg))
            res = asyncio.get_event_loop().run_until_complete(handler_fn(publisher, job_data))

            msg = completed_msg_fn(job_data, res)

            client.put_job(json.encode(msg))
            client.delete_job(job.job_id)
        except BeanstalkError as err:
            print(f"Received error during execution: {err}")
            client.release_job(job.job_id)
        except NotFoundError as err:
            msg = failed_msg_fn(job_data, err)

            traceback.print_exc()
            client.put_job(json.encode(msg))
            client.delete_job(job.job_id)
        except Exception as err:
            # Unhandled Exception
            traceback.print_exc()
            client.release_job(job.job_id)
        finally:
            time.sleep(0.5)

class ProgressReporter(abc.ABC):
    def increment(self, count: int):
        """Increment the counter by processed variant count and report to the beanstalk queue"""
        pass
    def get_counter(self) -> int:
        """Get the current value of the counter"""
        pass

@ray.remote(num_cpus=0)
class BeanstalkdProgressReporter(ProgressReporter):
    """A Ray class to report progress to a beanstalk queue"""

    def __init__(self, publisher: ProgressPublisher):
        self._value = 0
        self._message = deepcopy(publisher.message)
        self._message.data = ProgressData()

        self._client = BeanstalkClient(publisher.host, publisher.port, socket_timeout=10)
        self._client.use(publisher.queue)

    def increment(self, count: int):
        """Increment the counter by processed variant count and report to the beanstalk queue"""
        self._value += count
        self._message.data.progress = self._value # type: ignore

        self._client.put_job(json.encode(self._message))

    def get_counter(self):
        """Get the current value of the counter"""
        return self._value


@ray.remote(num_cpus=0)
class DebugProgressReporter(ProgressReporter):
    """A Ray class to report progress to stdout"""

    def __init__(self):
        self._value = 0

    def increment(self, count: int):
        self._value += count
        print(f"Processed {self._value} records")

    def get_counter(self):
        return self._value

def get_progress_reporter(publisher: ProgressPublisher | None) -> ObjectRef:
    if publisher:
        return BeanstalkdProgressReporter.remote(ProgressPublisher)  # type: ignore # pylint: disable=no-member

    return DebugProgressReporter.remote()  # type: ignore # pylint: disable=no-member