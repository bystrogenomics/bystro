"""
    CLI tool to start PRS job listener
"""
import argparse
from ruamel.yaml import YAML

from bystro.beanstalkd.worker import (
    QueueConf,
    listen,
)
from bystro.beanstalkd.messages import SubmittedJobMessage
from bystro.prs.messages import PRSJobData, PRSJobResult, PRSJobResultMessage
from bystro.prs.handler import calculate_prs_scores

TUBE = "prs"


def submit_msg_fn(job_data: PRSJobData):
    return SubmittedJobMessage(submission_id=job_data.submission_id)


def completed_msg_fn(job_data: PRSJobData, results: PRSJobResult) -> PRSJobResultMessage:
    return PRSJobResultMessage(submission_id=job_data.submission_id, results=results)


def main():
    """
    Start PRS listener that handles PRS jobs
    """
    parser = argparse.ArgumentParser(description=f"Start a listener for {TUBE} Bystro jobs")

    parser.add_argument(
        "--queue_conf",
        type=str,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
        required=True,
    )

    args = parser.parse_args()

    with open(args.queue_conf, "r", encoding="utf-8") as queue_config_file:
        queue_conf = YAML(typ="safe").load(queue_config_file)

    listen(
        job_data_type=PRSJobData,
        handler_fn=calculate_prs_scores,
        submit_msg_fn=submit_msg_fn,
        completed_msg_fn=completed_msg_fn,
        queue_conf=QueueConf(**queue_conf["beanstalkd"]),
        tube=TUBE,
    )


if __name__ == "__main__":
    main()
