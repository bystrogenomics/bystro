"""A module to handle the saving of search results into a new annotation"""

# TODO 2023-05-08: Track number of skipped entries
# TODO 2023-05-08: Write to Arrow IPC or Parquet, alongside tsv.
# TODO 2023-05-08: Implement distributed pipeline/filters/transforms
# TODO 2023-05-08: Support sort queries
# TODO 2023-05-08: get max_slices from opensearch index settings
# TODO 2023-05-08: concatenate chunks in a different ray worker

import logging
import math
import os
import psutil
import pathlib
import subprocess
import time

from numpy._typing import NDArray

import numpy as np
from opensearchpy import OpenSearch, AsyncOpenSearch

import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
import pyarrow.dataset as ds  # type: ignore
import pyarrow.ipc as ipc  # type: ignore
import ray

from bystro.beanstalkd.worker import ProgressPublisher, get_progress_reporter
from bystro.search.utils.annotation import AnnotationOutputs
from bystro.search.utils.messages import SaveJobData
from bystro.search.utils.opensearch import gather_opensearch_args
from bystro.utils.compress import get_compress_from_pipe_cmd, get_decompress_to_pipe_cmd
from bystro.utils.timer import Timer

logger = logging.getLogger(__name__)

MAX_QUERY_SIZE = 10_000
MAX_SLICES = 1e6
KEEP_ALIVE = "1d"
MAX_CONCURRENCY_PER_THREAD = 4
SAVE_FILTER_BATCH_READAHEAD = int(os.getenv("SAVE_FILTER_BATCH_READAHEAD", 0))
SAVE_FILTER_BATCH_READ_SIZE = int(os.getenv("SAVE_FILTER_BATCH_READ_SIZE", 200_000))

# How many scroll requests for each worker to handle
PARALLEL_SCROLL_CHUNK_INCREMENT = 2
# Percentage of fetched records to report progress after
REPORTING_INTERVAL = 0.1
MINIMUM_RECORDS_TO_ENABLE_REPORTING = 10_000
# These are the fields that are required to define a locus
# They are used to filter the dosage matrix
FIELDS_TO_QUERY = ["chrom", "pos", "inputRef", "alt"]

ray.init(ignore_reinit_error=True, address="auto")


def _clean_query(input_query_body: dict):
    if "sort" in input_query_body:
        del input_query_body["sort"]

    if "aggs" in input_query_body:
        del input_query_body["aggs"]

    if "slice" in input_query_body:
        del input_query_body["slice"]

    if "size" in input_query_body:
        del input_query_body["size"]

    if "track_total_hits" in input_query_body:
        del input_query_body["track_total_hits"]

    return input_query_body


@ray.remote
class AsyncQueryProcessor:
    REPORT_INCREMENT = 40_000

    def __init__(self, search_client_args: dict, reporter):
        # Initialize the async OpenSearch client during actor construction
        self.client = AsyncOpenSearch(**search_client_args)
        self.last_reported_count = 0
        self.reporter = reporter

    async def process_query(self, query: dict) -> tuple[NDArray[np.uint32] | None, list[str] | None]:
        doc_ids = []
        loci = []

        # Perform the search operation asynchronously using the pre-initialized client
        resp = await self.client.search(**query)

        for doc in resp["hits"]["hits"]:
            doc_ids.append(int(doc["_id"]))

            src = doc["fields"]
            loci.append(f"{src['chrom'][0]}:{src['pos'][0]}:{src['inputRef'][0]}:{src['alt'][0]}")

        if len(loci) == 0:
            return None, None

        if self.last_reported_count > self.REPORT_INCREMENT:
            self.reporter.increment.remote(self.last_reported_count)
            self.last_reported_count = 0

        self.last_reported_count += len(loci)

        return np.array(doc_ids), loci

    def close(self):
        if self.last_reported_count > 0:
            self.reporter.increment.remote(self.last_reported_count)


def _get_num_slices(client, index_name, max_query_size, max_slices, query):
    """Count number of hits for the index"""
    query_no_sort = query.copy()
    if "sort" in query_no_sort:
        del query_no_sort["sort"]
    if "track_total_hits" in query_no_sort:
        del query_no_sort["track_total_hits"]

    response = client.count(body=query_no_sort, index=index_name)

    n_docs = response["count"]

    if n_docs == 0:
        raise RuntimeError("No documents found for the query")

    # Opensearch does not always query the requested number of documents, and
    # we have observed up to 3% loss; to be safe, assume 15% max and then round
    # number of slices requested up
    expected_query_size_with_loss = max_query_size * 0.85

    num_slices_required = math.ceil(n_docs / expected_query_size_with_loss)
    if num_slices_required > max_slices:
        raise RuntimeError(
            "Too many slices required to process the query. Please reduce the query size."
        )

    return max(num_slices_required, 1)


def _prepare_query_body(query, slice_id, num_slices):
    """Prepare the query body for the slice"""
    body = query.copy()
    body["slice"] = {"id": slice_id, "max": num_slices}
    return body


def _correct_loci_case(loci):
    corrected_loci = []
    for locus in loci:
        # Check and replace the prefix as needed
        if locus.startswith("chrx"):
            corrected_loci.append("chrX" + locus[4:])
        elif locus.startswith("chry"):
            corrected_loci.append("chrY" + locus[4:])
        elif locus.startswith("chrm"):
            corrected_loci.append("chrM" + locus[4:])
        elif locus.startswith("chrmt"):
            corrected_loci.append("chrMT" + locus[5:])
        else:
            corrected_loci.append(locus)
    return corrected_loci


def run_dosage_filter(
    parent_dosage_matrix_path: str,
    dosage_out_path: str,
    loci_path: str,
    queue_config_path: str,
    progress_frequency: int,
    submission_id: str,
    binary_path: str = "dosage-filter",
):
    """
    Run the dosage_filter binary

    Args:
        binary_path (str): The path to the binary executable.
        args (list[str]): The list of arguments to pass to the binary.

    Returns:
        None

    Raises:
        RuntimeError: If the binary execution fails or if there is an error in the stderr output.
    """

    # call the dosage-filter program, which takes an --input --output --loci --progress-frequency --queue-config --job-submission-id args
    # and filters the dosage matrix to only include the loci in the loci file
    dosage_filter_cmd = (
        f"{binary_path} --input {parent_dosage_matrix_path} --output {dosage_out_path} "
        f"--loci {loci_path} --progress-frequency {progress_frequency} --queue-config {queue_config_path} "
        f"--job-submission-id {submission_id}"
    )

    logger.info("Beginning to filter genotypes")

    # Run the command and capture stderr
    process = subprocess.Popen(dosage_filter_cmd, stderr=subprocess.PIPE)
    _, stderr = process.communicate()

    if process.returncode != 0 or stderr:
        raise RuntimeError(f"Binary execution failed: {stderr.decode('utf-8')}")

    return


async def go(  # pylint:disable=invalid-name
    job_data: SaveJobData, search_conf: dict, publisher: ProgressPublisher, queue_config_path: str
) -> AnnotationOutputs:
    """Main function for running the query and writing the output"""
    output_dir = os.path.dirname(job_data.output_base_path)
    basename = os.path.basename(job_data.output_base_path)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    outputs, stats = AnnotationOutputs.from_path(
        output_dir, basename, job_data.input_file_names.config, compress=True
    )

    parent_dosage_matrix_path = os.path.join(
        job_data.input_dir, job_data.input_file_names.dosage_matrix_out_path
    )
    parent_annotation_path = os.path.join(job_data.input_dir, job_data.input_file_names.annotation)

    dosage_out_path = os.path.join(output_dir, outputs.dosage_matrix_out_path)

    search_client_args = gather_opensearch_args(search_conf)
    client = OpenSearch(**search_client_args)

    query = _clean_query(job_data.query_body)
    num_slices = _get_num_slices(client, job_data.index_name, MAX_QUERY_SIZE, MAX_SLICES, query)
    pit_id = client.create_point_in_time(index=job_data.index_name, params={"keep_alive": KEEP_ALIVE})["pit_id"]  # type: ignore   # noqa: E501

    reporter = get_progress_reporter(publisher)
    query["pit"] = {"id": pit_id}
    query["size"] = MAX_QUERY_SIZE
    query["fields"] = FIELDS_TO_QUERY
    query["_source"] = False

    try:
        num_cpus = ray.available_resources().get("CPU", 1)

        actor_index = 0
        reqs = []

        # slice api requires more than 1 slice
        actor_constructor = AsyncQueryProcessor.options(  # type: ignore
            max_concurrency=MAX_CONCURRENCY_PER_THREAD
        )

        with Timer() as timer:

            if num_slices == 1:
                actor = actor_constructor.remote(search_client_args, reporter)  # type: ignore

                results = ray.get([actor.process_query.remote({"body": query})])

                # Report any remaining rows
                ray.get(actor.close.remote())
            else:
                actors = []
                for _ in range(int(num_cpus)):
                    actors.append(actor_constructor.remote(search_client_args, reporter))  # type: ignore

                for slice_id in range(num_slices):

                    # Prepare the query body for this specific slice
                    body = {"body": _prepare_query_body(query, slice_id, num_slices)}

                    # Select an actor for this task in a round-robin fashion
                    actor = actors[actor_index]

                    # Call process_query for this slice
                    res = actor.process_query.remote(body)
                    reqs.append(res)

                    # Move to the next actor for the next slice
                    actor_index = (actor_index + 1) % len(actors)

                results = ray.get(reqs)
                ray.get([actor.close.remote() for actor in actors])

        logger.info("Querying took %s seconds", timer.elapsed_time)

        with Timer() as timer:
            # Process and aggregate results from all actors...
            doc_ids = []
            loci = []
            n_hits = 0
            for doc_ids_chunk, loci_chunk in results:
                if doc_ids_chunk is None:
                    continue

                n_hits += doc_ids_chunk.shape[0]
                doc_ids.append(doc_ids_chunk)
                loci.extend(loci_chunk)

            doc_ids_nd = np.concatenate(doc_ids)
            doc_ids_nd.sort()

        logger.info("Concatenating document ids took %s seconds", timer.elapsed_time)

        logger.info(
            "Memory usage before genotype filtering: %s (MB)",
            psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        )

        annotation_path = os.path.join(output_dir, outputs.annotation)

        reporter.message.remote("About to filter dosage matrix and annotation tsv.gz.")  # type: ignore

        reporting_interval = np.max(
            math.ceil(n_hits * REPORTING_INTERVAL), MINIMUM_RECORDS_TO_ENABLE_REPORTING
        )

        reporter.message.remote(  # type: ignore
            f"Reporting filtering progress every {reporting_interval} records."
        )

        # Filter the dosage matrix, if it has rows
        if os.path.exists(parent_dosage_matrix_path) and os.stat(parent_dosage_matrix_path).st_size > 0:
            loci = _correct_loci_case(loci)
            # Write requested loci to disk for logging purposes
            with open(os.path.join(output_dir, f"{basename}_loci.txt"), "w") as loci_fh:
                for locus in loci:
                    loci_fh.write(locus + "\n")

            run_dosage_filter(
                parent_dosage_matrix_path=parent_dosage_matrix_path,
                dosage_out_path=dosage_out_path,
                loci_path=loci_fh.name,
                queue_config_path=queue_config_path,
                progress_frequency=reporting_interval,
                submission_id=str(job_data.submission_id),
            )

        with Timer() as timer:
            bgzip_cmd = get_compress_from_pipe_cmd(annotation_path)
            bgzip_decompress_cmd = get_decompress_to_pipe_cmd(parent_annotation_path)
            bystro_stats_cmd = stats.stdin_cli_stats_command

            filters = None

            with (
                subprocess.Popen(bystro_stats_cmd, shell=True, stdin=subprocess.PIPE) as stats,
                subprocess.Popen(bgzip_cmd, shell=True, stdin=subprocess.PIPE) as p,
                subprocess.Popen(bgzip_decompress_cmd, shell=True, stdout=subprocess.PIPE) as in_fh,
            ):
                if in_fh.stdout is None:
                    raise IOError("Failed to open annotation file for reading.")

                if p.stdin is None:
                    raise IOError("Failed to open filtered annotation file for writing.")

                if stats.stdin is None:
                    raise IOError("Failed to open stats file for writing.")

                i = -1
                current_target_index = 0
                header_written = False
                header_fields = None

                for line in iter(in_fh.stdout.readline, b""):
                    if not header_written:
                        p.stdin.write(line)
                        stats.stdin.write(line)
                        header_written = True

                        if filters is not None:
                            header_fields = line.rstrip().split(b"\t")

                            if job_data.pipeline is not None and len(job_data.pipeline) > 0:
                                filters = []
                                for filter_msg in job_data.pipeline:
                                    filter_fn = filter_msg.make_filter(header_fields)

                                    if filter_fn is not None:
                                        filters.append(filter_fn)
                        continue

                    i += 1
                    if i == doc_ids_nd[current_target_index]:
                        filtered = False
                        if filters is not None:
                            src = line.rstrip().split(b"\t")
                            for filter_fn in filters:
                                if filter_fn(src):
                                    filtered = True
                                    break

                        if not filtered:
                            if (
                                current_target_index > 0
                                and current_target_index % reporting_interval == 0
                            ):
                                reporter.message.remote(  # type: ignore
                                    (
                                        "Annotation/stats: Filtered & wrote "
                                        f"{current_target_index} of {n_hits} rows."
                                    )
                                )

                            p.stdin.write(line)
                            stats.stdin.write(line)

                        # Move to the next target line number, if any
                        current_target_index += 1

                        if current_target_index >= n_hits:
                            reporter.message.remote(  # type: ignore
                                (
                                    "Annotation/stats: Filtered & wrote "
                                    f"{current_target_index} of {n_hits} rows."
                                )
                            )
                            reporter.message.remote("Done, cleaning up.")  # type: ignore
                            break

                p.stdin.close()  # Close the stdin to signal that we're done sending input
                stats.stdin.close()  # Close the stdin to signal that we're done sending input

                p.wait()
                stats.wait()
        logger.info("Filtering annotation and stats took %s seconds", timer.elapsed_time)
    except Exception as e:
        # Handle exceptions and cleanup, including deleting the PIT ID
        client.delete_point_in_time(body={"pit_id": pit_id})
        raise e

    client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore

    return outputs
