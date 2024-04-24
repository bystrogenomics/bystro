"""A module to handle the saving of search results into a new annotation"""

# TODO 2023-05-08: Track number of skipped entries
# TODO 2023-05-08: Implement distributed pipeline/filters/transforms
# TODO 2023-05-08: Support sort queries
# TODO 2023-05-08: get max_slices from opensearch index settings

import logging
import math
import os
import psutil
import pathlib
import subprocess
from typing import Callable

from opensearchpy import OpenSearch, AsyncOpenSearch

from numpy.typing import NDArray
import numpy as np
import ray

from bystro.beanstalkd.worker import ProgressPublisher, ProgressReporter, get_progress_reporter
from bystro.search.utils.annotation import AnnotationOutputs, Statistics
from bystro.search.utils.messages import SaveJobData
from bystro.search.utils.opensearch import gather_opensearch_args
from bystro.utils.compress import get_compress_from_pipe_cmd, get_decompress_to_pipe_cmd
from bystro.utils.timer import Timer


logger = logging.getLogger(__name__)

MAX_QUERY_SIZE = 10_000
MAX_SLICES = 1e6
KEEP_ALIVE = "1d"
MAX_CONCURRENCY_PER_THREAD = 4
SAVE_LOCI_BATCH_WRITE_SIZE = int(os.getenv("SAVE_LOCI_BATCH_WRITE_SIZE", 500_000))
# Annotations are reported by filtered input row;
# this is very rapid, so to avoid spamming the user with messages,
# we only report every 5 million rows by default
ANNOTATION_MINIMUM_REPORTING_INTERVAL = int(
    os.getenv("ANNOTATION_MINIMUM_REPORTING_INTERVAL", 5_000_0000)
)

# How many scroll requests for each worker to handle
PARALLEL_SCROLL_CHUNK_INCREMENT = 2
# Percentage of fetched records to report progress after
REPORTING_INTERVAL = 0.01
MINIMUM_RECORDS_TO_ENABLE_REPORTING = 100_000
# These are the fields that are required to define a locus
# They are used to filter the dosage matrix
FIELDS_TO_QUERY = ["chrom", "pos", "inputRef", "alt"]

ray.init(ignore_reinit_error=True, address="auto")

_GO_HANDLER_BINARY_PATH = "dosage-filter"


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

    async def process_query(self, query: dict) -> tuple[NDArray[np.int32], NDArray] | None:
        doc_ids: list[int] = []
        loci: list[str] = []

        # Perform the search operation asynchronously using the pre-initialized client
        resp = await self.client.search(**query)

        for doc in resp["hits"]["hits"]:
            src = doc["fields"]
            doc_id = int(doc["_id"])
            locus = f"{upper_chr(src['chrom'][0])}:{src['pos'][0]}:{src['inputRef'][0]}:{src['alt'][0]}"
            doc_ids.append(doc_id)
            loci.append(locus)

        if len(doc_ids) == 0:
            return None

        if self.last_reported_count > self.REPORT_INCREMENT:
            self.reporter.increment_and_write_progress_message.remote(  # type: ignore
                self.last_reported_count, "Fetched", "variants"
            )
            self.last_reported_count = 0

        self.last_reported_count += len(doc_ids)

        return np.array(doc_ids, dtype=np.int32), np.array(loci, dtype=object)

    def close(self):
        if self.last_reported_count > 0:
            self.reporter.increment_and_write_progress_message.remote(  # type: ignore
                self.last_reported_count, "Fetched", "variants"
            )
            self.last_reported_count = 0


def _get_num_slices(client, index_name, max_query_size, max_slices, query) -> tuple[int, int]:
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

    return max(num_slices_required, 1), n_docs


def _prepare_query_body(query, slice_id, num_slices):
    """Prepare the query body for the slice"""
    body = query.copy()
    body["slice"] = {"id": slice_id, "max": num_slices}
    return body


def upper_chr(chrom: str):
    return chrom[0:3] + chrom[3:].upper()


def run_dosage_filter(
    parent_dosage_matrix_path: str,
    dosage_out_path: str,
    loci_path: str,
    queue_config_path: str,
    progress_frequency: int,
    submission_id: str,
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

    # call the dosage-filter program, which takes an --input --output --loci
    # --progress-frequency --queue-config --job-submission-id args
    # and filters the dosage matrix to only include the loci in the loci file
    dosage_filter_cmd = (
        f"{_GO_HANDLER_BINARY_PATH} --input {parent_dosage_matrix_path} --output {dosage_out_path} "
        f"--loci {loci_path} --progress-frequency {progress_frequency} "
        f"--queue-config {queue_config_path} "
        f"--job-submission-id {submission_id}"
    )

    logger.info("Beginning to filter genotypes using command `%s`", dosage_filter_cmd)

    # Run the command and capture stderr
    process = subprocess.Popen(dosage_filter_cmd, stderr=subprocess.PIPE, shell=True, text=True)
    _, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Binary execution failed: {stderr}")

    return


def sort_loci_and_doc_ids(
    results: list[tuple[NDArray[np.int32], NDArray] | None]
) -> tuple[NDArray[np.int32], NDArray, int]:
    # Initialize empty numpy arrays for document loci and IDs
    all_loci = np.array([], dtype=np.int32)
    all_doc_ids = np.array([], dtype=object)

    # Aggregate results from all actors
    n_hits = 0
    for chunk in results:
        if chunk is None:
            continue

        doc_ids, loci = chunk
        n_hits += len(doc_ids)
        all_loci = np.concatenate((all_loci, loci))
        all_doc_ids = np.concatenate((all_doc_ids, doc_ids))

    # Get the indices that would sort the loci array
    sorted_indices = np.argsort(all_doc_ids)

    # Perform in-place sorting by reassigning sorted values back into the original arrays
    all_loci = all_loci[sorted_indices]
    all_doc_ids = all_doc_ids[sorted_indices]

    return all_doc_ids, all_loci, n_hits


def filter_annotation(
    stats: Statistics,
    annotation_path: str,
    parent_annotation_path: str,
    job_data: SaveJobData,
    doc_ids_sorted: NDArray[np.int32],
    loci_sorted: NDArray,
    n_hits: int,
    reporter: ProgressReporter,
    reporting_interval: int,
    loci_file_path: str,
):
    reporting_interval = max(ANNOTATION_MINIMUM_REPORTING_INTERVAL, reporting_interval)

    reporter.message.remote(  # type: ignore
        (
            "Filtering annotation file and re-generating stats. "
            f"Reporting progress every ~{reporting_interval} rows."
        )
    )

    filtered_loci = ""
    n_retained = 0
    with Timer() as timer:
        bgzip_cmd = get_compress_from_pipe_cmd(annotation_path)
        bgzip_decompress_cmd = get_decompress_to_pipe_cmd(parent_annotation_path)
        bystro_stats_cmd = stats.stdin_cli_stats_command

        with (
            subprocess.Popen(bystro_stats_cmd, shell=True, stdin=subprocess.PIPE) as stats_fh,
            subprocess.Popen(bgzip_cmd, shell=True, stdin=subprocess.PIPE) as p,
            subprocess.Popen(bgzip_decompress_cmd, shell=True, stdout=subprocess.PIPE) as in_fh,
            open(loci_file_path, "w") as loci_fh,
        ):
            if in_fh.stdout is None:
                raise IOError("Failed to open annotation file for reading.")

            if p.stdin is None:
                raise IOError("Failed to open filtered annotation file for writing.")

            if stats_fh.stdin is None:
                raise IOError("Failed to open stats file for writing.")

            i = -1
            current_target_index = 0
            header_written = False
            header_fields = None
            filters: list[Callable[[list[bytes]], bool]] = []

            for line in iter(in_fh.stdout.readline, b""):
                if not header_written:
                    p.stdin.write(line)
                    stats_fh.stdin.write(line)
                    header_written = True

                    if job_data.pipeline is not None and len(job_data.pipeline) > 0:
                        header_fields = line.rstrip().split(b"\t")
                        filters = []
                        for filter_msg in job_data.pipeline:
                            filter_fn = filter_msg.make_filter(header_fields)

                            if filter_fn is not None:
                                filters.append(filter_fn)
                    continue

                i += 1

                # doc_ids_sorted is a sorted list of document_ids, which are the indices in the
                # annotation file that we wish to keep
                # we iterate over input file lines (indexed by i), finding those lines that
                # are in doc_ids_sorted
                # because doc_ids_sorted is sorted in ascending order,
                # and we are reading the file from the beginning, we are guaranteed
                # that if i != doc_ids_sorted[current_target_index]
                # i is less than doc_ids_sorted[current_target_index]
                # or we have reached the end of doc_ids_sorted

                if i == doc_ids_sorted[current_target_index]:
                    filtered = False
                    if len(filters) > 0:
                        src = line.rstrip().split(b"\t")

                        for filter_fn in filters:
                            if filter_fn(src):
                                filtered = True
                                break

                    if not filtered:
                        n_retained += 1

                        p.stdin.write(line)
                        stats_fh.stdin.write(line)

                        filtered_loci += loci_sorted[current_target_index] + "\n"

                        if current_target_index % SAVE_LOCI_BATCH_WRITE_SIZE == 0:
                            loci_fh.write(filtered_loci)
                            filtered_loci = ""

                    # Move to the next target line number, if any
                    current_target_index += 1

                    if current_target_index >= n_hits:
                        break

                if i > 0 and i % reporting_interval == 0:
                    reporter.message.remote(   # type: ignore
                        f"Annotation: Filtered {i} rows. {current_target_index} survived filtering."
                    )

            if len(filtered_loci) > 0:
                loci_fh.write(filtered_loci)
                filtered_loci = ""

            loci_fh.close()
            p.stdin.close()  # Close the stdin to signal that we're done sending input
            stats_fh.stdin.close()  # Close the stdin to signal that we're done sending input

            p.wait()
            stats_fh.wait()

    reporter.message.remote(f"Annotation: {n_retained} variants survived filtering.")  # type: ignore

    reporter.message.remote("Annotation: Completed filtering.")  # type: ignore

    logger.info("Filtering annotation and generating stats took %s seconds", timer.elapsed_time)

    return n_retained


def filter_dosage_matrix(
    dosage_out_path: str,
    parent_dosage_matrix_path: str,
    job_data: SaveJobData,
    reporter: ProgressReporter,
    queue_config_path: str,
    reporting_interval: int,
    loci_file_path: str,
):
    if not (
        os.path.exists(parent_dosage_matrix_path) and os.stat(parent_dosage_matrix_path).st_size > 0
    ):
        logger.info("No dosage matrix to filter")
        reporter.message.remote("No dosage matrix to filter.")  # type: ignore
        # Touch the output file to avoid errors downstream
        pathlib.Path(dosage_out_path).touch()
        return

    reporter.message.remote(  # type: ignore
        f"Filtering dosage matrix file. Reporting progress every ~{reporting_interval} variants"
    )

    with Timer() as timer:
        run_dosage_filter(
            parent_dosage_matrix_path=parent_dosage_matrix_path,
            dosage_out_path=dosage_out_path,
            loci_path=loci_file_path,
            queue_config_path=queue_config_path,
            progress_frequency=reporting_interval,
            submission_id=str(job_data.submission_id),
        )
    logger.info("Filtering dosage matrix took %s seconds", timer.elapsed_time)


def filter_annotation_and_dosage_matrix(
    job_data: SaveJobData,
    reporter: ProgressReporter,
    doc_ids_sorted: NDArray[np.int32],
    loci_sorted: NDArray,
    n_hits: int,
    queue_config_path: str,
) -> AnnotationOutputs:
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

    annotation_path = os.path.join(output_dir, outputs.annotation)

    reporting_interval = max(MINIMUM_RECORDS_TO_ENABLE_REPORTING, math.ceil(n_hits * REPORTING_INTERVAL))

    loci_file_path = os.path.join(output_dir, f"{basename}_loci.txt")

    logger.info(
        "Memory usage before filter_annotation: %s (MB)",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    )

    n_results = filter_annotation(
        stats=stats,
        annotation_path=annotation_path,
        parent_annotation_path=parent_annotation_path,
        job_data=job_data,
        doc_ids_sorted=doc_ids_sorted,
        loci_sorted=loci_sorted,
        n_hits=n_hits,
        reporter=reporter,
        reporting_interval=reporting_interval,
        loci_file_path=loci_file_path,
    )

    del doc_ids_sorted

    logger.info(
        "Memory usage after filter_annotation: %s (MB)",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    )

    filter_dosage_matrix(
        dosage_out_path=dosage_out_path,
        parent_dosage_matrix_path=parent_dosage_matrix_path,
        loci_file_path=loci_file_path,
        job_data=job_data,
        reporter=reporter,
        queue_config_path=queue_config_path,
        reporting_interval=reporting_interval,
    )

    logger.info(
        "Memory usage after filter_dosage_matrix: %s (MB)",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    )

    reporter.increment.remote(n_results, True)  # type: ignore

    return outputs


async def go(  # pylint:disable=invalid-name
    job_data: SaveJobData, search_conf: dict, publisher: ProgressPublisher, queue_config_path: str
) -> AnnotationOutputs:
    """Main function for running the query and writing the output"""
    search_client_args = gather_opensearch_args(search_conf)
    client = OpenSearch(**search_client_args)

    query = _clean_query(job_data.query_body)
    num_slices, num_docs = _get_num_slices(
        client, job_data.index_name, MAX_QUERY_SIZE, MAX_SLICES, query
    )
    pit_id = client.create_point_in_time(index=job_data.index_name, params={"keep_alive": KEEP_ALIVE})["pit_id"]  # type: ignore   # noqa: E501

    update_interval = max(MINIMUM_RECORDS_TO_ENABLE_REPORTING, math.ceil(num_docs * REPORTING_INTERVAL))
    reporter = get_progress_reporter(publisher, update_interval)

    reporter.message.remote(  # type: ignore
        f"Fetching variants from search engine, Reporting progress every ~{update_interval} variants."
    )

    query["pit"] = {"id": pit_id}
    query["size"] = MAX_QUERY_SIZE
    query["fields"] = FIELDS_TO_QUERY
    query["_source"] = False

    logger.info(
        "Memory usage before querying: %s (MB)",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    )

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
    finally:
        # Cleanup the PIT ID
        client.delete_point_in_time(body={"pit_id": pit_id})
        client.close()

    logger.info("Querying took %s seconds", timer.elapsed_time)

    logger.info(
        "Memory usage before query result sorting: %s (MB)",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    )

    doc_ids_sorted, loci_sorted, n_hits = sort_loci_and_doc_ids(results)

    logger.info(
        "Memory usage after query result sorting: %s (MB)",
        psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    )

    reporter.increment_and_write_progress_message.remote(  # type: ignore
        0, "Fetched", "variants", force=True
    )
    reporter.clear_progress.remote()  # type: ignore

    outputs = filter_annotation_and_dosage_matrix(
        job_data=job_data,
        reporter=reporter,
        doc_ids_sorted=doc_ids_sorted,
        loci_sorted=loci_sorted,
        n_hits=n_hits,
        queue_config_path=queue_config_path,
    )

    return outputs
