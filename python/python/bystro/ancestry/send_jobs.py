import logging
import sys
import time

from pystalk import BeanstalkClient

from bystro.ancestry.ancestry_types import AncestrySubmission
from bystro.ancestry.beanstalk import BeanstalkSubmissionMessage

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

client = BeanstalkClient("127.0.0.1", 11300)
client.use("ancestry")
client.watch("ancestry_events")

ancestry_request = AncestrySubmission(vcf_path="foo.vcf")
beanstalk_msg = BeanstalkSubmissionMessage(queue_id="1337", submission_id="1338", data=ancestry_request)
payload = beanstalk_msg.json()

logging.debug("sending submission: %s", payload)
client.put_job(payload)
logging.debug("waiting...")
time.sleep(1)

jobs_received = 0
for job in client.reserve_iter():
    jobs_received += 1
    try:
        logging.debug("got job: %s", job)
    except Exception:
        logging.exception("encountered exception while reserving %s", job)
        client.release_job(job.job_id)
        raise
    logging.debug("deleting job")
    client.delete_job(job.job_id)
logging.debug("received %s jobs", jobs_received)
