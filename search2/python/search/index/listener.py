"""
    CLI tool to start search indexing server that listens to beanstalkd queue
    and indexes submitted data in Opensearch
"""
import argparse
import os
from typing import Optional, Any

from msgspec import Struct
from ruamel.yaml import YAML

from search.index.handler import go
from search.utils.messages import BaseMessage, ProgressPublisher, QueueConf, get_config_file_path, listen
from search.utils.annotation import AnnotationOutputs

class IndexJobData(BaseMessage, frozen=True):
    inputDir: str
    inputFileNames: AnnotationOutputs
    indexName: str
    assembly: str
    fieldNames: list[str] | None = None
    indexConfig: dict | None = None

class IndexResultData(BaseMessage):
    indexConfig: dict
    fieldNames: list

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
        m_path = get_config_file_path(conf_dir, beanstalkd_job_data.assembly, '.mapping.y*ml')

        with open(m_path, 'r', encoding='utf-8') as f:
            mapping_conf = YAML(typ="safe").load(f)

        inputs = beanstalkd_job_data.inputFileNames

        if not inputs.archived:
            raise ValueError('Indexing currently only works for indexing archived (tarballed) results')

        tar_path = os.path.join(beanstalkd_job_data.inputDir, inputs.archived)

        return go(index_name=beanstalkd_job_data.indexName,
                  tar_path=tar_path,
                  mapping_conf=mapping_conf,
                  search_conf=search_conf,
                  publisher=publisher)

    def submit_msg_fn(base_msg: dict, input_job_details: IndexJobData): # pylint: disable=unused-argument
        return base_msg

    def completed_msg_fn(base_msg: dict, input_job_details: IndexJobData, results: Any):
        m_path = get_config_file_path(conf_dir, input_job_details.assembly, '.mapping.y*ml')

        with open(m_path, 'r', encoding='utf-8') as f:
            mapping_conf = YAML(typ="safe").load(f)

        return IndexJobData({**base_msg, "indexConfig": mapping_conf, "fieldNames": results}

    listen(handler_fn=handler_fn,
           submit_msg_fn=submit_msg_fn,
           completed_msg_fn=completed_msg_fn,
           queue_conf=QueueConf(**queue_conf["beanstalkd"]),
           tube='index')

if __name__ == "__main__":
    main()
