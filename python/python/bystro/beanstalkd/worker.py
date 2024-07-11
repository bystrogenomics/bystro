"""TODO: Add description here"""

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import signal
import sys
import time
import traceback
from collections.abc import Callable
from textwrap import dedent
from typing import Any, TypeVar


from msgspec import DecodeError, ValidationError, json
from pystalk import BeanstalkClient, BeanstalkError  # type: ignore

from bystro.beanstalkd.messages import (
    BeanstalkJobID,
    BaseMessage,
    FailedJobMessage,
    InvalidJobMessage,
    ProgressPublisher,
    QueueConf,
    ProgressMessage
)

executor = ThreadPoolExecutor(max_workers=1)

BEANSTALK_ERR_TIMEOUT = "TIMED_OUT"
QUEUE_RESERVE_JOB_TIMEOUT_TIME = int(os.getenv("BYSTRO_BEANSTALKD_RESERVE_JOB_TIMEOUT", "20"))
QUEUE_JOB_HEARTBEAT_INTERVAL = int(os.getenv("BYSTRO_BEANSTALKD_JOB_HEARTBEAT_INTERVAL", "20"))
# Must be larger than QUEUE_RESERVE_JOB_TIMEOUT_TIME and QUEUE_JOB_HEARTBEAT_INTERVAL
QUEUE_CONSUMER_SOCKET_TIMEOUT = int(os.getenv("BYSTRO_BEANSTALKD_CONSUMER_SOCKET_TIMEOUT", "30"))

T = TypeVar("T", bound=BaseMessage)
T2 = TypeVar("T2", bound=BaseMessage)
T3 = TypeVar("T3", bound=BaseMessage)

logger = logging.getLogger(__name__)


# Signal handler function
def sigterm_handler(_signum, _frame):
    print("SIGTERM received. Cleaning up...")
    executor.shutdown(wait=False)
    exit(0)


# Set up the signal handler in the main thread
signal.signal(signal.SIGTERM, sigterm_handler)


def handle_job(handler_fn, publisher, job_data):
    try:
        return handler_fn(publisher, job_data)
    except Exception as e:
        return e


def default_failed_msg_fn(
    job_data: T | None, job_id: BeanstalkJobID, err: Exception
) -> FailedJobMessage | InvalidJobMessage:  # noqa: E501
    """Default failed message function"""
    if job_data is None:
        return InvalidJobMessage(queue_id=job_id, reason=str(err))
    return FailedJobMessage(submission_id=job_data.submission_id, reason=str(err))


def _touch(client: BeanstalkClient, job_id: str | int):
    # Ping the server periodically while waiting for the handler to finish
    try:
        with client._sock_ctx() as socket:  # noqa: SLF001
            client._send_message(f"touch {job_id}", socket)  # noqa: SLF001
            body = client._receive_word(socket, b"TOUCHED", b"NOT_FOUND")  # noqa: SLF001
            if body == b"NOT_FOUND":
                logger.warning("Job %s not found", job_id)
            logger.debug("Touched job %s", job_id)
    except Exception as e:
        logger.exception("Ping error while waiting for handler: %s", e)


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
    hosts, ports = queue_conf.split_host_port()
    tube_conf = queue_conf.tubes[tube]
    client_confs = tuple(
        {"host": host, "port": port, "socket_timeout": QUEUE_CONSUMER_SOCKET_TIMEOUT}
        for (host, port) in zip(hosts, ports)
    )

    i = 0
    while True:
        i += 1
        job = None
        job_id = None
        job_data = None
        client = None
        try:
            offset = i % len(hosts)
            client_conf = client_confs[offset]

            client = BeanstalkClient(**client_conf)
            client.watch(tube_conf["submission"])
            client.use(tube_conf["events"])

            job = client.reserve_job(QUEUE_RESERVE_JOB_TIMEOUT_TIME)
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
                assert job_data is not None

                publisher = ProgressPublisher(
                    host=client.host,
                    port=client.port,
                    queue=tube_conf["events"],
                    message=ProgressMessage(submission_id=job_data.submission_id),
                )

                client.put_job(json.encode(submit_msg_fn(job_data)))

                # Submit the job to the ThreadPoolExecutor
                future = executor.submit(handle_job, handler_fn, publisher, job_data)

                # Ensure job is kept alive indefinitely, until completion
                # Some jobs are potentially weeks long
                last_touch_time = time.time()
                while True:
                    # Check if the handle_job task is complete
                    if future.done():
                        _touch(client, job_id)
                        last_touch_time = time.time()
                        break

                    if time.time() - last_touch_time >= QUEUE_JOB_HEARTBEAT_INTERVAL:
                        _touch(client, job_id)
                        last_touch_time = time.time()

                    time.sleep(1)

                res = future.result()

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
                continue

            traceback.print_exc()

            if client is None:
                print("Couldn't connect to Beanstalkd server, sleeping for 5s", file=sys.stderr)
                time.sleep(5)
                continue

            if job is None:
                print("Couldn't reserve job, sleeping for 1s", file=sys.stderr)
                time.sleep(1)
                continue

            client.release_job(job.job_id)
            time.sleep(1)
            continue
