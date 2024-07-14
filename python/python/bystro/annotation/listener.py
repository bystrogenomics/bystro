"""
    CLI tool to start annotation server that listens to beanstalkd queue
    and processes submitted data using the bystro-annotate script
"""

import argparse
import os
import glob
import tempfile
import json
import shutil
import yaml
from bystro.beanstalkd.messages import BaseMessage, CompletedJobMessage, SubmittedJobMessage

from msgspec import Struct
from bystro.beanstalkd.worker import listen, ProgressPublisher, QueueConf

TUBE = "annotation"
ANNOTATE_SCRIPT_NAME = "bystro-annotate.pl"


class AnnotationResults(Struct, frozen=True, forbid_unknown_fields=True, rename="camel"):
    output_file_names: dict
    total_annotated: int
    total_skipped: int


class AnnotationCompletedJobMessage(
    CompletedJobMessage, frozen=True, forbid_unknown_fields=True, rename="camel", kw_only=True
):
    results: AnnotationResults


class AnnotationJobData(BaseMessage, frozen=True, forbid_unknown_fields=True, rename="camel"):
    assembly: str
    input_file_path: list[str]
    options: dict
    output_base_path: str


def _run_annotation(json_config_file: str, result_summary_path: str) -> AnnotationResults:
    # Find the ANNOTATE_SCRIPT, either because it's in the path, or 
    # because it is found in ../../../../perl/bin
    if shutil.which(ANNOTATE_SCRIPT_NAME):
        script = ANNOTATE_SCRIPT_NAME
    else:
        script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../perl/bin", ANNOTATE_SCRIPT_NAME)
        )

        if not os.path.exists(script):
            raise FileNotFoundError(f"Could not find the bystro-annotate script at {script}")

    command = [
        "perl",
        script,
        "--json_config",
        json_config_file,
        "--result_summary_path",
        result_summary_path,
    ]

    returncode = os.system(" ".join(command))

    if returncode != 0:
        raise RuntimeError(f"Annotation script execution failed: {returncode}")

    with open(result_summary_path, "r", encoding="utf-8") as file:
        result = json.load(file)

    if not result:
        raise RuntimeError("Annotation failed: No results returned")

    if result.get("error"):
        raise RuntimeError(f"Annotation failed: {result.get('error')}")

    return AnnotationResults(
        output_file_names=result.get("results"),
        total_annotated=result.get("totalProgress"),
        total_skipped=result.get("totalSkipped"),
    )


def _get_config_file_path(assembly: str, conf_dir: str) -> str:
    pattern = os.path.join(conf_dir, f"{assembly}.y*ml")
    config_files = glob.glob(pattern)
    if not config_files:
        raise FileNotFoundError(f"No config path found for the assembly {assembly}")
    return config_files[0]


def handler_fn(
    publisher: ProgressPublisher, job_data: AnnotationJobData, conf_dir: str
) -> AnnotationResults:
    input_data = {
        "input_files": job_data.input_file_path,
        "output_file_base": job_data.output_base_path,
        "config": _get_config_file_path(job_data.assembly, conf_dir),
        "options": job_data.options,
        "compress": 1,
        "assembly": job_data.assembly,
        "run_statistics": 1,
        "archive": 0,
        "publisher": {
            "messageBase": {
                "event": "progress",
                "submissionId": job_data.submission_id,
                "data": None,
            },
            "queue": publisher.queue,
            "server": f"{publisher.host}:{publisher.port}",
        },
    }

    try:
        with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as json_config_file:
            json.dump(input_data, json_config_file)
            json_config_file_path = json_config_file.name

        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", encoding="utf-8"
        ) as result_summary_file:
            result_summary_path = result_summary_file.name

        return _run_annotation(json_config_file_path, result_summary_path)
    finally:
        if json_config_file_path and os.path.exists(json_config_file_path):
            os.remove(json_config_file_path)
        if result_summary_path and os.path.exists(result_summary_path):
            os.remove(result_summary_path)


def completed_msg_fn(job_data: AnnotationJobData, result_data: AnnotationResults):
    return AnnotationCompletedJobMessage(submission_id=job_data.submission_id, results=result_data)


def submit_msg_fn(job_data: AnnotationJobData):
    return SubmittedJobMessage(job_data.submission_id)


def main():
    parser = argparse.ArgumentParser(
        description="Start annotation server that listens to beanstalkd queue"
    )
    parser.add_argument(
        "--conf_dir", type=str, help="Path to the genome/assembly config directory", required=True
    )
    parser.add_argument(
        "--queue_conf", type=str, help="Path to the beanstalkd queue config yaml file", required=True
    )
    args = parser.parse_args()

    conf_dir = args.conf_dir
    queue_conf_path = args.queue_conf

    with open(queue_conf_path, "r", encoding="utf-8") as file:
        queue_conf = yaml.safe_load(file)

    listen(
        job_data_type=AnnotationJobData,
        handler_fn=lambda publisher, job_data: handler_fn(publisher, job_data, conf_dir),
        submit_msg_fn=submit_msg_fn,
        completed_msg_fn=completed_msg_fn,
        queue_conf=QueueConf(**queue_conf["beanstalkd"]),
        tube=TUBE,
    )

if __name__ == "__main__":
    main()