"""Integration test listener."""

import json
import logging
import sys
import time

from cattrs import unstructure
from cattrs.preconf.json import make_converter
from pystalk import BeanstalkClient

from bystro.ancestry.ancestry_types import AncestrySubmission

json_converter = make_converter()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

client = BeanstalkClient("127.0.0.1", 11300)
client.use("ancestry")
client.watch("ancestry_events")

ancestry_submission = AncestrySubmission(vcf_path="foo.vcf")

sabot = json.dumps({"submissionID": 1337, "ancestry_submission": unstructure(ancestry_submission)})
logging.debug("sending submission: %s", sabot)
client.put_job(sabot)
logging.debug("waiting for 5 seconds...")
time.sleep(5)

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
