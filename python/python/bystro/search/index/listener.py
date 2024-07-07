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

from msgspec import json, Struct

TUBE = "index"

_INDEXER_BINARY = "opensearch"


class IndexerReturnData(Struct, frozen=True, forbid_unknown_fields=True, rename="camel"):
    """
    The return data from the indexer binary.
    """

    header: list[str]
    total_indexed: int


def run_binary_with_args(binary_path: str, args: list[str]) -> IndexerReturnData:
    """
    Run a binary with the specified arguments and return the output as a list of strings.

    Args:
        binary_path (str): The path to the binary executable.
        args (list[str]): The list of arguments to pass to the binary.

    Returns:
        list[str]: The output of the binary as a list of strings.

    Raises:
        RuntimeError: If the binary execution fails or if there is an error in the stderr output.
    """

    # Construct the command
    command = " ".join([binary_path] + args)

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0 or stderr:
        raise RuntimeError(f"Binary execution failed: {stderr}")

    return_data = stdout

    return json.decode(return_data, type=IndexerReturnData)


def run_handler_with_config(
    index_name: str,
    mapping_config: str,
    opensearch_config: str,
    annotation_path: str,
    no_queue: bool = False,
    submission_id: SubmissionID | None = None,
    queue_config: str | None = None,
) -> IndexerReturnData:
    """
    Run the handler with the specified configuration.

    Args:
        index_name (str): The name of the index.
        mapping_config (str): The path to the mapping configuration file.
        opensearch_config (str): The path to the OpenSearch configuration file.
        annotation_path (str): The path to the annotation file.
        no_queue (bool, optional): Whether to disable the queue. Defaults to False.
        submission_id (SubmissionID | None, optional):
            The submission ID. Required when no_queue is not False (optional).
        queue_config (str | None, optional):
            The path to the queue configuration file. Required when no_queue is not False (optional).

    Returns:
        IndexerReturnData: The header fields.

    Raises:
        ValueError: If submission_id and queue_config are not specified when no_queue is not False.
    """

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

    return run_binary_with_args(_INDEXER_BINARY, args)


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

    def handler_fn(_: ProgressPublisher, beanstalkd_job_data: IndexJobData) -> IndexerReturnData:
        inputs = beanstalkd_job_data.input_file_names

        annotation_path = os.path.join(beanstalkd_job_data.input_dir, inputs.annotation)
        m_path = get_config_file_path(conf_dir, beanstalkd_job_data.assembly, ".mapping.y*ml")

        return run_handler_with_config(
            index_name=beanstalkd_job_data.index_name,
            submission_id=beanstalkd_job_data.submission_id,
            mapping_config=m_path,
            opensearch_config=search_conf,
            queue_config=queue_conf,
            annotation_path=annotation_path,
        )

    def submit_msg_fn(job_data: IndexJobData):
        return SubmittedJobMessage(job_data.submission_id)

    def completed_msg_fn(job_data: IndexJobData, return_data: IndexerReturnData):
        mapping_config_path = get_config_file_path(conf_dir, job_data.assembly, ".mapping.y*ml")

        # Write mapping config path to the job data out_dir directory
        map_config_basename = os.path.basename(mapping_config_path)
        out_dir = job_data.out_dir
        map_config_out_path = os.path.join(out_dir, map_config_basename)
        shutil.copyfile(mapping_config_path, map_config_out_path)

        return IndexJobCompleteMessage(
            submission_id=job_data.submission_id,
            results=IndexJobResults(
                index_config_path=map_config_basename,
                field_names=return_data.header,
                total_indexed=return_data.total_indexed,
            ),
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
