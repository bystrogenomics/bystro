"""TODO: Add description here"""

import argparse
from typing import Any

from ruamel.yaml import YAML

from search.save.handler import go
from search.utils.beanstalkd import QueueConf, listen, Publisher, get_config_file_path

required_keys = ("outputBasePath", "assembly", "queryBody", "fieldNames", "indexName")
optional_keys = ("indexConfig", "pipeline")

def _coerce_inputs(
    job_details: dict,
):
    job_specific_args = {}

    for key in required_keys:
        if key not in job_details:
            raise ValueError(f"Missing required key: {key} in job message")

        job_specific_args[key] = job_details[key]

    for key in optional_keys:
        if key in job_details:
            job_specific_args[key] = job_details[key]

    return {
        "compress": True,
        "archive": True,
        "run_statistics": True,
        **job_specific_args
    }

def main():
    """TODO: Docstring for main."""
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

    config_path_base_dir = args.conf_dir
    with open(args.queue_conf, "r", encoding="utf-8") as queue_config_file:
        queue_conf = YAML(typ="safe").load(queue_config_file)

    with open(args.search_conf, "r", encoding="utf-8") as search_config_file:
        search_conf = YAML(typ="safe").load(search_config_file)

    def handler_fn(publisher: Publisher, job_details: dict):
        input_body = _coerce_inputs(job_details)
        return go(input_body=input_body, search_conf=search_conf, publisher=publisher)

    def submit_msg_fn(base_msg: dict, job_details: dict):
        config_path = get_config_file_path(config_path_base_dir, job_details['assembly'])

        with open(config_path, 'r', encoding='utf-8') as f: # pylint: disable=invalid-name
            job_config = YAML(typ="safe").load(f)

        return {**base_msg, "jobConfig": job_config}

    def completed_msg_fn(base_msg: dict, job_details: dict, results: Any): # pylint: disable=unused-argument
        return {**base_msg, "results": {"outputFileNames": results}}

    listen(
        handler_fn=handler_fn,
        submit_msg_fn=submit_msg_fn,
        completed_msg_fn=completed_msg_fn,
        queue_conf=QueueConf(**queue_conf["beanstalkd"]),
        tube="saveFromQuery",
    )

if __name__ == "__main__":
    main()
