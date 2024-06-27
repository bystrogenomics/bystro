"""TODO: Add description here"""

import abc
import queue
import sys
import time
import threading
import traceback
from collections.abc import Callable
from textwrap import dedent
from typing import Any, TypeVar

import ray
from msgspec import DecodeError, Struct, ValidationError, json
from pystalk import BeanstalkClient, BeanstalkError  # type: ignore
from pystalk.client import Job  # type: ignore

from bystro.beanstalkd.messages import (
    BeanstalkJobID,
    BaseMessage,
    FailedJobMessage,
    InvalidJobMessage,
    ProgressMessage,
    ProgressStringMessage,
)

BEANSTALK_ERR_TIMEOUT = "TIMED_OUT"
SOCKET_TIMEOUT_TIME = 10
JOB_TIMEOUT_TIME = 5
HEARTBEAT_INTERVAL = 30 # seconds

T = TypeVar("T", bound=BaseMessage)
T2 = TypeVar("T2", bound=BaseMessage)
T3 = TypeVar("T3", bound=BaseMessage)


class ProgressPublisher(Struct):
    """Beanstalkd Message Published Config"""

    host: str
    port: int
    queue: str
    message: ProgressMessage


class QueueConf(Struct):
    """Queue Configuration"""

    addresses: list[str]
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


def default_failed_msg_fn(
    job_data: T | None, job_id: BeanstalkJobID, err: Exception
) -> FailedJobMessage | InvalidJobMessage:  # noqa: E501
    """Default failed message function"""
    if job_data is None:
        return InvalidJobMessage(queue_id=job_id, reason=str(err))
    return FailedJobMessage(submission_id=job_data.submission_id, reason=str(err))


def handle_job(publisher, job_data, handler_fn, result_queue):
    try:
        result = handler_fn(publisher, job_data)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)


def listen(
    job_data_type: type[T],
    handler_fn: Callable[[ProgressPublisher, T], Any],
    submit_msg_fn: Callable[[T], T2],
    completed_msg_fn: Callable[[T, Any], T3],
    queue_conf: QueueConf,
    tube: str,
    failed_msg_fn: Callable[
        [T | None, BeanstalkJobID, Exception], FailedJobMessage | InvalidJobMessage
    ] = default_failed_msg_fn,  # noqa: E501
):
    """Listen on a Beanstalkd channel, waiting for work.
    When work is available call the work handler
    """
    hosts, ports = queue_conf.split_host_port()

    tube_conf = queue_conf.tubes[tube]
    client_confs = tuple(
        {"host":host, "port":port, "socket_timeout":SOCKET_TIMEOUT_TIME}
        for (host, port) in zip(hosts, ports)
    )

    i = 0
    while True:
        i += 1

        job: Job | None = None
        job_id: BeanstalkJobID | None = None
        job_data: T | None = None
        client: BeanstalkClient | None = None
        try:
            offset = i % len(hosts)
            client_conf = client_confs[offset]

            client = BeanstalkClient(**client_conf)
            client.watch(tube_conf["submission"])
            client.use(tube_conf["events"])

            job = client.reserve_job(JOB_TIMEOUT_TIME)
            job_id = int(job.job_id)  # type: ignore

            try:
                job_data = json.decode(job.job_data, type=job_data_type)
            except ValidationError as err:
                msg = dedent(
                    f"""
                            Job {job_id} JSON does not have the data expected.
                            Expected {job_data_type.keys_with_types()}.
                            Decoding failed with: `{err}`"""
                )
                traceback.print_exc()
                client.put_job(json.encode(failed_msg_fn(job_data, job_id, ValueError(msg))))
                client.delete_job(job.job_id)
                continue
            except DecodeError as err:
                msg = dedent(
                    f"""
                            Job {job.job_id} JSON is invalid.
                            Decoding `{str(job.job_data)}`, failed with: `{err}`"""
                )
                traceback.print_exc()
                client.put_job(json.encode(failed_msg_fn(job_data, job_id, ValueError(msg))))
                client.delete_job(job_id)
                continue
            except Exception:
                traceback.print_exc()

                client.put_job(
                    json.encode(
                        failed_msg_fn(job_data, job_id, Exception("Unknown error, check admin logs"))
                    )
                )
                client.delete_job(job_id)

            try:
                # Typeguard
                assert job_data is not None

                publisher = ProgressPublisher(
                    host=client.host,
                    port=client.port,
                    queue=tube_conf["events"],
                    message=ProgressMessage(submission_id=job_data.submission_id),
                )

                client.put_job(json.encode(submit_msg_fn(job_data)))

                result_queue: queue.Queue = queue.Queue()
                handler_thread = threading.Thread(
                    target=handle_job, args=(publisher, job_data, handler_fn, result_queue)
                )
                handler_thread.start()

                while handler_thread.is_alive():
                    # Ping the server periodically while waiting for the handler to finish
                    try:
                        with client._sock_ctx() as socket:  # noqa: SLF001
                            client._send_message(f"touch {job_id}", socket)  # noqa: SLF001
                            body = client._receive_word(socket, b"TOUCHED", b"NOT_FOUND")  # noqa: SLF001
                            if body == b"NOT_FOUND":
                                print(f"Job {job_id} not found", file=sys.stderr)
                    except Exception as e:
                        print(f"Ping error while waiting for handler: {e}", file=sys.stderr)
                    finally:
                        time.sleep(HEARTBEAT_INTERVAL)

                handler_thread.join()  # Ensure the handler thread has completed

                res = result_queue.get()
                if isinstance(res, Exception):
                    raise res

                client.put_job(json.encode(completed_msg_fn(job_data, res)))
                client.delete_job(job.job_id)
            except Exception as err:
                traceback.print_exc()

                failed_msg = failed_msg_fn(job_data, job_id, err)

                client.put_job(json.encode(failed_msg))
                client.delete_job(job.job_id)

                continue

        except BeanstalkError as err:
            if err.message == BEANSTALK_ERR_TIMEOUT:
                # This is completely expected, will happen every 5s
                continue

            traceback.print_exc()

            if client is None:
                print("Couldn't connect to Beanstalkd server, sleeping for 10s", file=sys.stderr)
                time.sleep(10)
                continue

            if job is None:
                print("Couldn't reserve job, sleeping for 10s", file=sys.stderr)
                time.sleep(10)
                continue

            client.release_job(job.job_id)

            time.sleep(1)
            continue


class ProgressReporter(abc.ABC):
    @abc.abstractmethod
    def increment(self, count: int, force: bool = False):
        """Increment the counter by processed variant count and report to the beanstalk queue"""

    @abc.abstractmethod
    def message(self, msg: str):
        """Send a message to the beanstalk queue"""

    @abc.abstractmethod
    def increment_and_write_progress_message(
        self, count: int, msg_prefix: str, msg_suffix: str = "", force: bool = False
    ):
        """Increment the counter by processed variant count
        and report to the beanstalk queue as a string message"""

    @abc.abstractmethod
    def clear_progress(self):
        """Clear the progress counter"""

    @abc.abstractmethod
    def get_counter(self) -> int:
        """Get the current value of the counter"""


@ray.remote(num_cpus=0)
class BeanstalkdProgressReporter(ProgressReporter):
    """A Ray class to report progress to a beanstalk queue"""

    def __init__(self, publisher: ProgressPublisher, update_interval: int = 100_000):
        self._message = publisher.message
        self._client = BeanstalkClient(publisher.host, publisher.port, socket_timeout=10)
        self._client.use(publisher.queue)
        self._update_interval = update_interval

        self._last_updated = 0

    def increment(self, count: int, force: bool = False):
        """Increment the counter by processed variant count and report to the beanstalk queue"""
        self._message.data.progress += count

        if force or self._message.data.progress - self._last_updated >= self._update_interval:
            self._client.put_job(json.encode(self._message))
            self._last_updated = self._message.data.progress

    def increment_and_write_progress_message(
        self, count: int, msg_prefix: str, msg_suffix: str = "", force: bool = False
    ):
        """Increment the counter by processed variant count
        and report to the beanstalk queue as a string message
        """
        self._message.data.progress += count

        if force or self._message.data.progress - self._last_updated >= self._update_interval:
            message = f"{msg_prefix} {self._message.data.progress} {msg_suffix}"
            progress_message = ProgressStringMessage(
                submission_id=self._message.submission_id, data=message
            )
            self._client.put_job(json.encode(progress_message))
            self._last_updated = self._message.data.progress

    def clear_progress(self):
        """Clear the progress counter"""
        self._message.data.progress = 0
        self._last_updated = 0

    def message(self, msg: str):
        """Send a message to the beanstalk queue"""
        progress_message = ProgressStringMessage(submission_id=self._message.submission_id, data=msg)
        self._client.put_job(json.encode(progress_message))

    def get_counter(self) -> int:
        """Get the current value of the counter"""
        return self._message.data.progress


@ray.remote(num_cpus=0)
class DebugProgressReporter(ProgressReporter):
    """A Ray class to report progress to stdout"""

    def __init__(self):
        self._value = 0

    def increment(self, count: int, _force: bool = False):
        self._value += count
        print(f"Processed {self._value} records")

    def increment_and_write_progress_message(
        self, count: int, msg_prefix: str, msg_suffix: str = "", _force: bool = False
    ):
        self._value += count
        print(f"{msg_prefix} {self._value} {msg_suffix}")

    def clear_progress(self):
        """Clear the progress counter"""
        self._value = 0

    def message(self, msg: str):
        """Send a message to the beanstalk queue"""
        print(msg)

    def get_counter(self):
        return self._value


def get_progress_reporter(
    publisher: ProgressPublisher | None = None, update_interval: int = 100_000
) -> ProgressReporter:
    if publisher:
        return BeanstalkdProgressReporter.remote(  # type: ignore
            publisher, update_interval=update_interval
        )

    return DebugProgressReporter.remote()  # type: ignore
