"""Provide helper classes for communicating over beanstalk."""
# this functionality is superceded now by bystro.beanstalkd and will be removed before this PR goes in.
import logging
import re
import urllib
from collections.abc import Callable
from enum import Enum
from ipaddress import AddressValueError, IPv4Address
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Extra, Field, validator
from pystalk import BeanstalkClient, BeanstalkError
from ruamel.yaml import YAML

logger = logging.getLogger()

MAX_VALID_PORT = 2**16 - 1


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as file_handler:
        return YAML(typ="safe").load(file_handler)


class Address(BaseModel, extra=Extra.forbid):
    """Model a host and port."""

    host: str
    port: int = Field(ge=0, le=MAX_VALID_PORT)

    @classmethod
    def from_str(cls: type["Address"], raw_address: str) -> "Address":
        """Parse an Address from a string."""
        # urlsplit needs an absolute url, so we need to make it absolute if not so already.
        if not re.match(r"(beanstalkd:)?//", raw_address):
            absolute_address = "//" + raw_address
        else:
            absolute_address = raw_address
        # mypy seems confused about urllib's attrs
        parsed_address = urllib.parse.urlsplit(absolute_address)  # type: ignore [attr-defined]
        if parsed_address.hostname is None:
            msg = f"Couldn't find hostname in {raw_address}"
            raise ValueError(msg)
        if parsed_address.port is None:
            msg = f"Couldn't find port in {raw_address}"
            raise ValueError(msg)
        return cls(
            host=parsed_address.hostname,
            port=parsed_address.port,
        )

    @validator("host")
    def _host_is_valid_ip_address(cls: "Address", raw_host: str) -> str:  # noqa: [N805]
        """Ensure host is a valid IPv4 address."""
        #  We really shouldn't be storing IP addresses as raw strings at all, but this usage is
        #  ubiquitous in libraries we need to work with.  A compromise is to validate the address
        #  string upon receipt and store it as str afterwards.
        try:
            host = IPv4Address(raw_host)
        except AddressValueError as err:
            err_msg = f"Couldn't parse host ({raw_host}) as IPv4Address: {err}"
            raise ValueError(err_msg) from err
        return str(host)


class BeanstalkEvent(Enum):
    """Represent a beanstalk event type."""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    PROGRESS = "progress"


class BeanstalkBaseMessage(BaseModel, extra=Extra.forbid):
    """Wrapper class for beanstalk messages."""

    queue_id: str
    submission_id: str
    data: Any


class BeanstalkSubmissionMessage(BeanstalkBaseMessage, extra=Extra.forbid):
    """Represent a submission, i.e. incoming message to a worker."""


class BeanstalkEventMessage(BeanstalkBaseMessage, extra=Extra.forbid):
    """Represent a beanstalk event, i.e. an outgoing response from a worker."""

    event: BeanstalkEvent


BEANSTALK_TIMEOUT_ERROR = "TIMED_OUT"


class BystroBeanstalkClient:
    def __init__(self, clients: list[BeanstalkClient]) -> None:
        self.clients = clients
        self.cur_client_idx = 0

    @classmethod
    def from_config(cls, queue_config_path: str, worker_name: str) -> "BystroBeanstalkClient":
        beanstalk_conf = _load_yaml(Path(queue_config_path))["beanstalkd"]
        addresses: dict[str, Any] = beanstalk_conf["addresses"]
        worker_tubes: dict[str, Any] = beanstalk_conf["tubes"][worker_name]
        submission_tube = worker_tubes["submission"]
        events_tube = worker_tubes["events"]

        clients = []
        for address in addresses:
            parsed_address = Address.from_str(address)
            client = BeanstalkClient(parsed_address.host, parsed_address.port)
            client.watchlist = {submission_tube}
            client.use(events_tube)
            logger.info(
                "starting beanstalk client on: %s using tube(s): %s",
                parsed_address,
                client.watchlist,
            )
            clients.append(client)
        return cls(clients)

    def handle_job(
        self, handler_fn: Callable[[BeanstalkSubmissionMessage], BeanstalkEventMessage]
    ) -> None:
        self.cur_client_idx += 1
        client = self.clients[self.cur_client_idx % len(self.clients)]
        try:
            job = client.reserve_job()
        except BeanstalkError as err:
            if err.message == BEANSTALK_TIMEOUT_ERROR:
                logger.debug("Timed out while reserving a job: this is expected if no jobs are present")
                return
            raise
        try:
            sub_msg = BeanstalkSubmissionMessage.parse_raw(job.job_data.decode())
            event_msg = handler_fn(sub_msg)
        except:
            logger.exception("Encountered exception while handling job %s", job)
            client.release_job(job.job_id)
            raise
        try:
            client.put_job(event_msg.json())
        except:
            logger.exception("Encountered exception while putting event job %s in queue", job)
            client.release_job(job.job_id)
            raise

        client.delete_job(job.job_id)
