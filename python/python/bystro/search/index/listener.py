"""
    CLI tool to start search indexing server that listens to beanstalkd queue
    and indexes submitted data in Opensearch
"""

import argparse
import os
import subprocess
import shutil

from ruamel.yaml import YAML

from bystro.beanstalkd.messages import SubmissionID, SubmittedJobMessage
from bystro.beanstalkd.worker import ProgressPublisher, QueueConf, listen
from bystro.search.utils.annotation import get_config_file_path
from bystro.search.utils.messages import IndexJobCompleteMessage, IndexJobData, IndexJobResults
from bystro.utils.config import _get_bystro_project_root

from msgspec import json

TUBE = "index"

_GO_HANDLER_BINARY_PATH = None


def get_go_handler_binary_path() -> str:
    """Return path to go handler binary."""
    global _GO_HANDLER_BINARY_PATH
    if _GO_HANDLER_BINARY_PATH is None:
        _GO_HANDLER_BINARY_PATH = os.path.join(
            _get_bystro_project_root(), "go/cmd/opensearch/opensearch"
        )

    if not os.path.exists(_GO_HANDLER_BINARY_PATH):
        raise ValueError(
            (
                f"Binary not found at {_GO_HANDLER_BINARY_PATH}. "
                "Please ensure to build the binary first, by running "
                "`go build` in the `go/cmd/opensearch` directory."
            )
        )

    return _GO_HANDLER_BINARY_PATH


def run_binary_with_args(binary_path: str, args: list[str]) -> list[str]:
    """
    Run the binary with specified arguments and handle errors.
    :param binary_path: Path to the binary file.
    :param args: List of arguments for the binary.
    :return: None
    """
    # Construct the command
    command = [binary_path] + args

    # Run the command and capture stderr
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0 or stderr:
        raise Exception(f"Binary execution failed: {stderr.decode('utf-8')}")

    headerFieldsStr = stdout.decode("utf-8")

    return json.decode(headerFieldsStr, type=list[str])


def run_handler_with_config(
    index_name: str,
    mapping_config: str,
    opensearch_config: str,
    annotation_path: str,
    no_queue: bool = False,
    submission_id: SubmissionID | None = None,
    queue_config: str | None = None,
) -> list[str]:
    binary_path = get_go_handler_binary_path()
    args = [
        "--index-name",
        index_name,
        "--mapping-config",
        mapping_config,
        "--opensearch-config",
        opensearch_config,
        "--input",
        annotation_path,
    ]

    if no_queue:
        args.append("--no-queue")
    else:
        if submission_id is None or queue_config is None:
            raise ValueError(
                "submission_id and queue_config must be specified when no_queue is not False"
            )

        args.append("--queue-config")
        args.append(queue_config)

        args.append("--job-submission-id")
        args.append(str(submission_id))

    header_fields = run_binary_with_args(binary_path, args)

    return header_fields


def main():
    """
    Start search indexing server that listens to beanstalkd queue
    and indexes submitted data in Opensearch
    """
    parser = argparse.ArgumentParser(description="Process some config files.")
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

    conf_dir = args.conf_dir
    queue_conf = args.queue_conf
    search_conf = args.search_conf

    with open(args.queue_conf, "r", encoding="utf-8") as queue_config_file:
        queue_conf_deserialized = YAML(typ="safe").load(queue_config_file)

    def handler_fn(_: ProgressPublisher, beanstalkd_job_data: IndexJobData) -> list[str]:
        inputs = beanstalkd_job_data.input_file_names

        annotation_path = os.path.join(beanstalkd_job_data.input_dir, inputs.annotation)
        m_path = get_config_file_path(conf_dir, beanstalkd_job_data.assembly, ".mapping.y*ml")

        header_fields = run_handler_with_config(
            index_name=beanstalkd_job_data.index_name,
            submission_id=beanstalkd_job_data.submission_id,
            mapping_config=m_path,
            opensearch_config=search_conf,
            queue_config=queue_conf,
            annotation_path=annotation_path,
        )

        return header_fields

    def submit_msg_fn(job_data: IndexJobData):
        return SubmittedJobMessage(job_data.submission_id)

    def completed_msg_fn(job_data: IndexJobData, field_names: list[str]):
        mapping_config_path = get_config_file_path(conf_dir, job_data.assembly, ".mapping.y*ml")

        # Write mapping config path to the job data out_dir directory
        map_config_basename = os.path.basename(mapping_config_path)
        out_dir = job_data.out_dir
        map_config_out_path = os.path.join(out_dir, map_config_basename)
        shutil.copyfile(mapping_config_path, map_config_out_path)

        return IndexJobCompleteMessage(
            submission_id=job_data.submission_id,
            results=IndexJobResults(map_config_basename, field_names),
        )  # noqa: E501

    listen(
        job_data_type=IndexJobData,
        handler_fn=handler_fn,
        submit_msg_fn=submit_msg_fn,
        completed_msg_fn=completed_msg_fn,
        queue_conf=QueueConf(**queue_conf_deserialized["beanstalkd"]),
        tube=TUBE,
    )


if __name__ == "__main__":
    main()
