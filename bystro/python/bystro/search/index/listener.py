"""
    CLI tool to start search indexing server that listens to beanstalkd queue
    and indexes submitted data in Opensearch
"""
import argparse
import os

from ruamel.yaml import YAML

from bystro.search.index.handler import go
from bystro.search.utils.messages import (
    IndexJobData,
    IndexJobResults,
    IndexJobCompleteMessage
)
from bystro.search.utils.beanstalkd import ProgressPublisher, QueueConf, get_config_file_path, listen

TUBE = "index"


def main():
    """
    Start search indexing server that listens to beanstalkd queue
    and indexes submitted data in Opensearch
    """
    parser = argparse.ArgumentParser(description="Process some config files.")
    parser.add_argument(
        "--conf_dir", type=str, help="Path to the genome/assembly config directory"
    )
    parser.add_argument(
        "--queue_conf",
        type=str,
        help="Path to the beanstalkd queue config yaml file (e.g beanstalk1.yml)",
    )
    parser.add_argument(
        "--search_conf",
        type=str,
        help="Path to the opensearch config yaml file (e.g. elasticsearch.yml)",
    )
    args = parser.parse_args()

    conf_dir = args.conf_dir
    with open(args.queue_conf, "r", encoding="utf-8") as queue_config_file:
        queue_conf = YAML(typ="safe").load(queue_config_file)

    with open(args.search_conf, "r", encoding="utf-8") as search_config_file:
        search_conf = YAML(typ="safe").load(search_config_file)

    def handler_fn(publisher: ProgressPublisher, beanstalkd_job_data: IndexJobData):
        m_path = get_config_file_path(conf_dir, beanstalkd_job_data.assembly, ".mapping.y*ml")

        with open(m_path, "r", encoding="utf-8") as f:
            mapping_conf = YAML(typ="safe").load(f)

        inputs = beanstalkd_job_data.inputFileNames

        if not inputs.archived:
            raise ValueError(
                "Indexing currently only works for indexing archived (tarballed) results"
            )

        tar_path = os.path.join(beanstalkd_job_data.inputDir, inputs.archived)

        return go(
            index_name=beanstalkd_job_data.indexName,
            tar_path=tar_path,
            mapping_conf=mapping_conf,
            search_conf=search_conf,
            publisher=publisher,
        )

    def submit_msg_fn(job_data: IndexJobData):
        return job_data

    def completed_msg_fn(job_data: IndexJobData, fieldNames: list[str]):
        m_path = get_config_file_path(conf_dir, job_data.assembly, ".mapping.y*ml")

        with open(m_path, "r", encoding="utf-8") as f:
            mapping_conf = YAML(typ="safe").load(f)

        return IndexJobCompleteMessage(job_data.submissionID, IndexJobResults(mapping_conf, fieldNames))

    listen(
        job_data_type=IndexJobData,
        handler_fn=handler_fn,
        submit_msg_fn=submit_msg_fn,
        completed_msg_fn=completed_msg_fn,
        queue_conf=QueueConf(**queue_conf["beanstalkd"]),
        tube=TUBE,
    )


if __name__ == "__main__":
    main()
