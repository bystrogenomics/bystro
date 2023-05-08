"""TODO: Add description here"""

import argparse

from ruamel.yaml import YAML

from search.save.handler import go
from search.utils.beanstalkd import QueueConf, listen

required_keys = ("outputBasePath", "assembly", "queryBody", "fieldNames", "indexName")
optional_keys = ("indexConfig", "pipeline")


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

    listen(
        fn=go,
        queue_conf=QueueConf(**queue_conf["beanstalkd"]),
        search_conf=search_conf,
        config_path_base_dir=config_path_base_dir,
        required_keys=required_keys,
        optional_keys=optional_keys,
        tube="saveFromQuery",
    )


if __name__ == "__main__":
    main()
