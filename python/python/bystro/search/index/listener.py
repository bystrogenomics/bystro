"""
    CLI tool to start search indexing server that listens to beanstalkd queue
    and indexes submitted data in Opensearch
"""
import argparse
import os
import subprocess

from ruamel.yaml import YAML

from bystro.beanstalkd.messages import SubmittedJobMessage
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
        inputs = beanstalkd_job_data.inputFileNames

        if not inputs.archived:
            raise ValueError("Indexing currently only works for indexing archived (tarballed) results")

        tar_path = os.path.join(beanstalkd_job_data.inputDir, inputs.archived)
        m_path = get_config_file_path(conf_dir, beanstalkd_job_data.assembly, ".mapping.y*ml")

        binary_path = get_go_handler_binary_path()
        args = [
            "--index-name",
            beanstalkd_job_data.indexName,
            "--job-submission-id",
            beanstalkd_job_data.submissionID,
            "--mapping-config",
            m_path,
            "--opensearch-config",
            search_conf,
            "--queue-config",
            queue_conf,
            "--tarball-path",
            tar_path,
        ]

        header_fields = run_binary_with_args(binary_path, args)

        return header_fields


    def submit_msg_fn(job_data: IndexJobData):
        return SubmittedJobMessage(job_data.submissionID)

    def completed_msg_fn(job_data: IndexJobData, fieldNames: list[str]):
        m_path = get_config_file_path(conf_dir, job_data.assembly, ".mapping.y*ml")

        with open(m_path, "r", encoding="utf-8") as f:
            mapping_conf = YAML(typ="safe").load(f)

        return IndexJobCompleteMessage(
            submissionID=job_data.submissionID, results=IndexJobResults(mapping_conf, fieldNames)
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
