import logging
import sys

from pystalk import BeanstalkClient

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

host, port = "127.0.0.1", 11300
client = BeanstalkClient(host, port)
client.watch("ancestry")

jobs_handled = 0
for job in client.reserve_iter():
    jobs_handled += 1
    logging.debug("got job: %s")
    logging.debug("deleting job")
    client.delete_job(job.job_id)
logging.debug("handled %s jobs", jobs_handled)
