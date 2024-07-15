"""TODO: Add description here"""

from concurrent import futures
from concurrent.futures import ProcessPoolExecutor

import logging
import multiprocessing
import os
import signal
import sys
import time
import traceback
from collections.abc import Callable
from textwrap import dedent
from typing import Any, TypeVar

import cloudpickle  # type: ignore
from msgspec import DecodeError, ValidationError, json
from pystalk import BeanstalkClient, BeanstalkError  # type: ignore
import psutil

from bystro.beanstalkd.messages import (
    BeanstalkJobID,
    BaseMessage,
    FailedJobMessage,
    InvalidJobMessage,
    ProgressPublisher,
    QueueConf,
    ProgressMessage,
)

BEANSTALK_ERR_TIMEOUT = "TIMED_OUT"
QUEUE_RESERVE_JOB_TIMEOUT_TIME = int(os.getenv("BYSTRO_BEANSTALKD_RESERVE_JOB_TIMEOUT", "20"))
QUEUE_JOB_HEARTBEAT_INTERVAL = int(os.getenv("BYSTRO_BEANSTALKD_JOB_HEARTBEAT_INTERVAL", "20"))
# Must be larger than QUEUE_RESERVE_JOB_TIMEOUT_TIME and QUEUE_JOB_HEARTBEAT_INTERVAL
QUEUE_CONSUMER_SOCKET_TIMEOUT = int(os.getenv("BYSTRO_BEANSTALKD_CONSUMER_SOCKET_TIMEOUT", "30"))

T = TypeVar("T", bound=BaseMessage)
T2 = TypeVar("T2", bound=BaseMessage)
T3 = TypeVar("T3", bound=BaseMessage)

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger()


class FunctionWrapper:
    def __init__(self, fn, queue):
        self.fn_ser = cloudpickle.dumps(fn)
        self.queue = queue

    def __call__(self, *args, **kwargs):
        worker_pid = os.getpid()
        self.queue.put(worker_pid)

        fn = cloudpickle.loads(self.fn_ser)
        return fn(*args, **kwargs)


# Signal handler function
def kill_child_processes(parent_pid=os.getpid(), kill_parent=False):
    parent = psutil.Process(parent_pid)

    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        try:
            logger.info("Killing child process %s", child)
            child.kill()
        except Exception as e:
            logger.error("Failed to kill child %s due to: %s", child, e)

    if kill_parent:
        try:
            logger.info("Killing parent process %s", parent)
            parent.kill()
        except Exception as e:
            logger.error("Failed to kill parent %s due to: %s", parent, e)


def sigterm_handler(_signum, _frame):
    kill_child_processes()
    sys.exit(0)


# Set up the signal handler in the main thread
signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigterm_handler)


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


def _touch(client: BeanstalkClient, job_id: BeanstalkJobID):
    # Ping the server periodically while waiting for the handler to finish
    with client._sock_ctx() as socket:  # noqa: SLF001
        client._send_message(f"touch {job_id}", socket)  # noqa: SLF001
        return client._receive_word(socket, b"TOUCHED")  # noqa: SLF001


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
    future: futures.Future | None = None
    while True:
        i += 1
        job = None
        job_id = None
        job_data = None
        client = None
        offset = i % len(hosts)
        client_conf = client_confs[offset]

        try:
            client = BeanstalkClient(**client_conf)
            client.watch(tube_conf["submission"])
            client.use(tube_conf["events"])
        except Exception as err:
            logger.error("Failed to connect to beanstalkd: %s", err)
            time.sleep(1)
            continue

        try:
            job = client.reserve_job(QUEUE_RESERVE_JOB_TIMEOUT_TIME)
            job_id = int(job.job_id)  # type: ignore
            logger.info("Reserved job with id: %s", job_id)
        except Exception as err:
            if isinstance(err, BeanstalkError) and err.message == BEANSTALK_ERR_TIMEOUT:
                logger.info("No jobs available")
            else:
                logger.error("Failed to reserve job: %s", err)
            time.sleep(1)
            continue

        try:
            job_data = json.decode(job.job_data, type=job_data_type)
            logger.info("Decoded data for job with id: %s: %s", job_id, job_data)
        except Exception as err:
            msg = Exception("Unknown error, check admin logs")

            if isinstance(err, DecodeError):
                msg = ValueError(
                    dedent(
                        f"""
                        Job {job_id} JSON is invalid.
                        Decoding `{str(job.job_data)}`, failed with: `{err}`"""
                    )
                )
            if isinstance(err, ValidationError):
                msg = ValueError(
                    dedent(
                        f"""
                        Job {job_id} JSON does not have the data expected.
                        Expected {job_data_type.keys_with_types()}.
                        Decoding failed with: `{err}`"""
                    )
                )

            client.put_job(json.encode(failed_msg_fn(job_data, job_id, ValueError(msg))))
            client.delete_job(job_id)

            traceback.print_exc()
            continue

        try:
            publisher = ProgressPublisher(
                host=client.host,
                port=client.port,
                queue=tube_conf["events"],
                message=ProgressMessage(submission_id=job_data.submission_id),
            )

            client.put_job(json.encode(submit_msg_fn(job_data)))

            with multiprocessing.Manager() as manager:
                pid_queue = manager.Queue()
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(FunctionWrapper(handler_fn, pid_queue), publisher, job_data)

                    future_pid = pid_queue.get()
                    logger.info(
                        "Started worker for function %s. Parent PID: %s, Worker pid: %s",
                        handler_fn,
                        os.getpid(),
                        future_pid,
                    )

                    # Ensure job is kept alive indefinitely, until completion
                    # Some jobs are potentially weeks long
                    last_touch_time = time.time()

                    failed_touch = False
                    while True:
                        failed_touch = False
                        time.sleep(1)
                        # Check if the handle_job task is complete
                        try:
                            if future.done():
                                _touch(client, job_id)
                                last_touch_time = time.time()
                                break

                            if time.time() - last_touch_time >= QUEUE_JOB_HEARTBEAT_INTERVAL:
                                _touch(client, job_id)
                                last_touch_time = time.time()
                        except TimeoutError:
                            logger.warning("Job %s  _touch timed out", job_id)
                        except Exception as err:
                            # The only expected error is NOT_FOUND which means the job is no longer bound
                            # to the worker.
                            # This means we must terminate processing of this job, since other
                            # workers may to pick it up for processing
                            # Any other exception is even more odd, and suggest a fatal error
                            # related to beanstalkd communication, which again could potentially
                            # lead to duplicate job processing
                            logger.error(
                                "Job %s _touch failed with %s, shutting down job processing",
                                job_id,
                                err,
                            )
                            failed_touch = True
                            break

                    if failed_touch:
                        executor.shutdown(wait=False, cancel_futures=True)

                        try:
                            kill_child_processes(future_pid, True)
                        except Exception as e:
                            logger.error("Failed to kill job worker PID: %s due to: %s", future_pid, e)

                        try:
                            client.release_job(job.job_id)
                        except Exception as err:
                            logger.error("Failed to release job with id %s due to: %s", job.job_id, err)

                        continue

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
