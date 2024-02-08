"""
    CLI tool to start search saving server that listens to beanstalkd queue
    and write submitted queries to disk as valid Bystro annotations
"""
import argparse

from ruamel.yaml import YAML

from bystro.beanstalkd.worker import (
    ProgressPublisher,
    QueueConf,
    listen,
)
from bystro.beanstalkd.messages import SubmittedJobMessage
from bystro.search.save.handler import go
from bystro.search.utils.annotation import AnnotationOutputs
from bystro.search.utils.messages import (
    SaveJobCompleteMessage,
    SaveJobData,
    SaveJobResults
)

TUBE = "saveFromQuery"


def main():
    """
    Start search saving server that listens to beanstalkd queue
    and write submitted queries to disk as valid Bystro annotations
    """
    parser = argparse.ArgumentParser(description=f"Start a listener for {TUBE} Bystro jobs")
    parser.add_argument(
        "--conf_dir", type=str, help="Path to the genome/assembly config directory", required=True
    )
    parser.add_argument(
        "--queue_conf",
        type=str,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
        required=True,
    )
    parser.add_argument(
        "--search_conf",
        type=str,
        help="Path to the opensearch config yaml file (e.g. elasticsearch.yml)",
        required=True,
    )
    args = parser.parse_args()

    with open(args.queue_conf, "r", encoding="utf-8") as queue_config_file:
        queue_conf = YAML(typ="safe").load(queue_config_file)

    with open(args.search_conf, "r", encoding="utf-8") as search_config_file:
        search_conf = YAML(typ="safe").load(search_config_file)

    def handler_fn(publisher: ProgressPublisher, job_data: SaveJobData):
        return go(job_data=job_data, search_conf=search_conf, publisher=publisher)

    def submit_msg_fn(job_data: SaveJobData):
        return SubmittedJobMessage(submission_id=job_data.submission_id)

    def completed_msg_fn(job_data: SaveJobData, results: AnnotationOutputs) -> SaveJobCompleteMessage:
        return SaveJobCompleteMessage(
            submission_id=job_data.submission_id, results=SaveJobResults(results)
        )

    listen(
        job_data_type=SaveJobData,
        handler_fn=handler_fn,
        submit_msg_fn=submit_msg_fn,
        completed_msg_fn=completed_msg_fn,
        queue_conf=QueueConf(**queue_conf["beanstalkd"]),
        tube=TUBE,
    )


if __name__ == "__main__":
    main()
