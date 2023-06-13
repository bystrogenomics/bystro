"""Integration test listener."""

import json
import logging
import sys
import time

from bystro.ancestry.ancestry_types import AncestrySubmission
from bystro.ancestry.listener import main as listener_main
from bystro.beanstalkd.worker import QueueConf
from cattrs import unstructure
from cattrs.preconf.json import make_converter
from pystalk import BeanstalkClient

json_converter = make_converter()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ANCESTRY_CHANNEL = "test_ancestry"
SUBMISSION_TUBE = "test_ancestry"
EVENTS_TUBE = "test_ancestry_events"
HOST = "127.0.0.1"
PORT = 11300  # TODO
ADDRESS = HOST + ":" + str(PORT)

client = BeanstalkClient(HOST, PORT)
client.use(SUBMISSION_TUBE)
client.watch(EVENTS_TUBE)


def send_job(client: BeanstalkClient):
    ancestry_submission = AncestrySubmission(vcf_path="foo.vcf")

    sabot = json.dumps({"submissionID": 1337, "ancestry_submission": unstructure(ancestry_submission)})
    logging.debug("sending submission: %s", sabot)
    client.put_job(sabot)
    logging.debug("waiting for 5 seconds...")
    time.sleep(5)


def receive_jobs(client: BeanstalkClient):
    jobs_received = 0
    for job in client.reserve_iter():
        jobs_received += 1
        try:
            logging.debug("got job: %s", job)
        except Exception:
            logging.exception("encountered exception while reserving %s", job)
            client.release_job(job.job_id)
            raise
        logging.debug("deleting job %s", job)
        client.delete_job(job.job_id)

    logging.debug("Ending, having received %s jobs", jobs_received)
    return jobs_received


def test_listener():
    # make sure beanstalk is up on a test port
    queue_conf = QueueConf(
        addresses=[ADDRESS],
        tubes={ANCESTRY_CHANNEL: {"submission": SUBMISSION_TUBE, "events": EVENTS_TUBE}},
    )
    print("about to listen")
    listener_main(queue_conf, SUBMISSION_TUBE)
    print("finished listening")
    send_job(client)
    jobs_received = receive_jobs(client)
    assert jobs_received == 2
