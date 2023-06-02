"""
    CLI tool to start search saving server that listens to beanstalkd queue
    and write submitted queries to disk as valid Bystro annotations
"""
import argparse
from typing import Any

from msgspec import Struct
from ruamel.yaml import YAML

from search.save.handler import go
from search.utils.beanstalkd import BaseMessage, FailedMessage, QueueConf, listen, ProgressPublisher, get_config_file_path
from search.utils.messages import SaveJobData, SaveJobSubmitMessage, SaveJobCompleteMessage

JOB_TYPE="saveFromQuery"

def main():
    """
        Start search saving server that listens to beanstalkd queue
        and write submitted queries to disk as valid Bystro annotations
    """
    parser = argparse.ArgumentParser(description=f"Start a listener for {JOB_TYPE} Bystro jobs")
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
        required=True
    )
    args = parser.parse_args()

    config_path_base_dir = args.conf_dir
    with open(args.queue_conf, "r", encoding="utf-8") as queue_config_file:
        queue_conf = YAML(typ="safe").load(queue_config_file)

    with open(args.search_conf, "r", encoding="utf-8") as search_config_file:
        search_conf = YAML(typ="safe").load(search_config_file)

    def handler_fn(publisher: ProgressPublisher, beanstalkd_job_data: SaveJobData):
        return go(job_data=beanstalkd_job_data, search_conf=search_conf, publisher=publisher)

    def submit_msg_fn(beanstalkd_job_data: SaveJobData):
        config_path = get_config_file_path(config_path_base_dir, beanstalkd_job_data.assembly)

        with open(config_path, 'r', encoding='utf-8') as f: # pylint: disable=invalid-name
            job_config = YAML(typ="safe").load(f)

        return SaveJobSubmitMessage(beanstalkd_job_data.submissionID, job_config)

    def completed_msg_fn(beanstalkd_job_data: SaveJobData, results: list[str]):
        return SaveJobCompleteMessage()(beanstalkd_job_data.submissionID, SaveJobCompleteMessage(results))

    def failed_msg_fn(beanstalkd_job_data: SaveJobData, exception: Exception):
        return FailedMessage(beanstalkd_job_data.submissionID, str(exception))

    listen(
        job_data_type=SaveJobCompleteMessage,
        handler_fn=handler_fn,
        submit_msg_fn=submit_msg_fn,
        completed_msg_fn=completed_msg_fn,
        failed_msg_fn=failed_msg_fn,
        queue_conf=QueueConf(**queue_conf["beanstalkd"]),
        tube="saveFromQuery",
    )

if __name__ == "__main__":
    main()
