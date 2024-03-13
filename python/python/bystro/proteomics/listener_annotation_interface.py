import argparse

import logging
from ruamel.yaml import YAML
from pathlib import Path
import pyarrow.dataset as ds    # type: ignore
from opensearchpy import OpenSearch
import pandas as pd
from datetime import datetime, timezone

from bystro.beanstalkd.worker import listen, QueueConf, ProgressPublisher
from bystro.beanstalkd.messages import BaseMessage, CompletedJobMessage, SubmittedJobMessage
from bystro.proteomics.annotation_interface import (
    get_annotation_result_from_query,
    join_annotation_result_to_proteomics_dataset,
)

logger = logging.getLogger(__file__)

PROTEOMICS_TUBE = "proteomics"


class ProteomicsJobData(BaseMessage, frozen=True, rename="camel"):
    """
    The expected JSON message for the Proteomics job.

    Parameters
    ----------
    data_path: str
        The path to the proteomics data file.
    out_dir: str
        The directory to write the results to.
    annotation_query: str
        The query string used to fetch annotation data from OpenSearch index.
    index_name: str
        The name of the OpenSearch index from which annotation data will be queried.
    """

    data_path: str
    out_dir: str
    annotation_query: str
    index_name: str


class ProteomicsJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True, rename="camel"):
    result_path: str


def _load_queue_conf(queue_conf_path: str) -> QueueConf:
    with Path(queue_conf_path).open(encoding="utf-8") as queue_config_file:
        raw_queue_conf = YAML(typ="safe").load(queue_config_file)
    beanstalk_conf = raw_queue_conf["beanstalkd"]
    return QueueConf(addresses=beanstalk_conf["addresses"], tubes=beanstalk_conf["tubes"])


def handler_fn(_publisher: ProgressPublisher, job_data: ProteomicsJobData) -> str:
    logger.info("Processing Proteomics job: %s", job_data)

    Path(job_data.out_dir).mkdir(parents=True, exist_ok=True)

    try:
        dataset = ds.dataset(job_data.data_path, format="arrow")
        gene_abundance_df = dataset.to_table().to_pandas()
    except Exception as arrow_exception:
        logger.exception(arrow_exception)
        try:
            gene_abundance_df = pd.read_csv(job_data.data_path, sep="\t")
        except Exception as pandas_exception:
            logger.exception(pandas_exception)
            raise ValueError(f"Failed to read {job_data.data_path}; not arrow feather or .tsv")

    client = OpenSearch()
    annotation_df = get_annotation_result_from_query(
        job_data.annotation_query, job_data.index_name, client
    )

    joined_df = join_annotation_result_to_proteomics_dataset(annotation_df, gene_abundance_df)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    result_path = Path(job_data.out_dir) / f"joined.{timestamp}.feather"
    joined_df.to_feather(result_path)

    logger.info("Proteomics job completed. Results saved to %s", result_path)
    return str(result_path)

def submit_msg_fn(proteomics_job_data: ProteomicsJobData) -> SubmittedJobMessage:
    logger.debug("Received ProteomicsJobData: %s", proteomics_job_data)
    return SubmittedJobMessage(proteomics_job_data.submission_id)


def completed_msg_fn(
    proteomics_job_data: ProteomicsJobData, result_path: str
) -> ProteomicsJobCompleteMessage:
    logger.debug("Proteomics job completed: %s, results at %s", proteomics_job_data, result_path)
    return ProteomicsJobCompleteMessage(
        submission_id=proteomics_job_data.submission_id, result_path=result_path
    )


def main(queue_conf: QueueConf) -> None:
    logger.info(
        "Proteomics worker is listening on addresses: %s, tube: %s...",
        queue_conf.addresses,
        PROTEOMICS_TUBE,
    )
    listen(
        ProteomicsJobData,
        handler_fn,
        submit_msg_fn,
        completed_msg_fn,
        queue_conf,
        PROTEOMICS_TUBE,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start proteomics data processing listener.")
    parser.add_argument(
        "--queue_conf",
        type=Path,
        help="Path to the beanstalkd queue config yaml file (e.g., beanstalk1.yml)",
        required=True,
    )
    args = parser.parse_args()

    queue_conf = _load_queue_conf(args.queue_conf)
    main(queue_conf)
