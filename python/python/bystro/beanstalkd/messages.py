import abc
from enum import Enum
import os
import time
from typing import get_type_hints

from msgspec import Struct, field, json
from pystalk import BeanstalkClient  # type: ignore
import ray

SubmissionID = str | int
BeanstalkJobID = int

# seconds; default AWS load balancer TTL is 60 seconds
# set the interval 2 seconds beforehand to make room for timing / network delays
QUEUE_HEARTBEAT_INTERVAL = max(1, float(os.getenv("BYSTRO_BEANSTALKD_HEARTBEAT_INTERVAL", "20")))
QUEUE_PRODUCER_TIMEOUT = max(1, float(os.getenv("BYSTRO_BEANSTALKD_PRODUCER_TIMEOUT", "30")))


class Event(str, Enum):
    """Beanstalkd Event"""

    PROGRESS = "progress"
    FAILED = "failed"
    STARTED = "started"
    COMPLETED = "completed"


class BaseMessage(Struct, frozen=True, rename="camel"):
    submission_id: SubmissionID

    @classmethod
    def keys_with_types(cls) -> dict:
        return get_type_hints(cls)


class SubmittedJobMessage(BaseMessage, frozen=True):
    event: Event = Event.STARTED


class CompletedJobMessage(BaseMessage, frozen=True):
    event: Event = Event.COMPLETED


class FailedJobMessage(BaseMessage, frozen=True):
    reason: str
    event: Event = Event.FAILED


class InvalidJobMessage(Struct, frozen=True, rename="camel"):
    # Invalid jobs that are invalid because the submission breaks serialization invariants
    # will not have a submission_id as that ID is held in the serialized data
    queue_id: BeanstalkJobID
    reason: str
    event: Event = Event.FAILED

    @classmethod
    def keys_with_types(cls) -> dict:
        return get_type_hints(cls)


class ProgressData(Struct):
    progress: int = 0
    skipped: int = 0


class ProgressMessage(BaseMessage, frozen=True):
    """Beanstalkd Message"""

    event: Event = Event.PROGRESS
    data: ProgressData = field(default_factory=ProgressData)


class ProgressStringMessage(BaseMessage, frozen=True):
    """Beanstalkd Message with a string progress"""

    data: str
    event: Event = Event.PROGRESS


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
        self._update_interval = update_interval
        self._publisher = publisher

        self._last_updated = 0
        self._last_update_time = time.time()

        self._client: BeanstalkClient | None = None

    def _get_client(self) -> BeanstalkClient:
        if self._client is None:
            self._client = BeanstalkClient(
                self._publisher.host, self._publisher.port, socket_timeout=QUEUE_PRODUCER_TIMEOUT
            )
            self._client.use(self._publisher.queue)
            self._last_update_time = time.time()
            return self._client

        # Will automatically be re-opened upon next use
        # Ensure we do not re-establish the connection more than every HEARTBEAT_INTERVAL seconds
        if time.time() - self._last_update_time >= QUEUE_HEARTBEAT_INTERVAL:
            print(
                (
                    "Set BeanstalkdProgressReporter client to closed."
                    " Will be automatically re-opened on next use"
                )
            )
            self._client.close()

        self._last_update_time = time.time()

        return self._client

    def increment(self, count: int, force: bool = False):
        """Increment the counter by processed variant count and report to the beanstalk queue"""
        self._message.data.progress += count

        if force or self._message.data.progress - self._last_updated >= self._update_interval:
            try:
                client = self._get_client()
                client.put_job(json.encode(self._message))
                self._last_updated = self._message.data.progress
                self._last_update_time = time.time()
            except Exception as e:
                print(f"Failed to put job to beanstalkd: {e}")

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
            try:
                client = self._get_client()
                client.put_job(json.encode(progress_message))

                self._last_updated = self._message.data.progress
                self._last_update_time = time.time()
            except Exception as e:
                print(f"Failed to put job to beanstalkd: {e}")

    def clear_progress(self):
        """Clear the progress counter"""
        self._message.data.progress = 0
        self._last_updated = 0

    def message(self, msg: str):
        """Send a message to the beanstalk queue"""
        progress_message = ProgressStringMessage(submission_id=self._message.submission_id, data=msg)

        try:
            client = self._get_client()
            client.put_job(json.encode(progress_message))
            self._last_update_time = time.time()
        except Exception as e:
            print(f"Failed to put job to beanstalkd: {e}")

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
