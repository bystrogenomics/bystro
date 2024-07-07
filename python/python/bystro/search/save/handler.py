"""A module to handle the saving of search results into a new annotation"""

# TODO 2023-05-08: Track number of skipped entries
# TODO 2023-05-08: Implement distributed pipeline/filters/transforms
# TODO 2023-05-08: Support sort queries
# TODO 2023-05-08: get max_slices from opensearch index settings

import gc
import logging
import math
import os
import psutil
import pathlib
import subprocess
import time
from typing import Callable

from opensearchpy import OpenSearch, AsyncOpenSearch

from numpy.typing import NDArray
import numpy as np
import ray

from bystro.beanstalkd.messages import get_progress_reporter, ProgressPublisher, ProgressReporter
from bystro.search.utils.annotation import AnnotationOutputs, Statistics
from bystro.search.utils.messages import SaveJobData, SaveJobResults
from bystro.search.utils.opensearch import gather_opensearch_args
from bystro.utils.compress import get_compress_from_pipe_cmd, get_decompress_to_pipe_cmd
from bystro.utils.timer import Timer


logger = logging.getLogger(__name__)

MAX_QUERY_SIZE = 10_000
# TODO 2024-04-26 @akotlar Investigate the impact of large numbers of slices
# and whether 20_000 slices is a reasonable limit
MAX_SLICES = 20_000
KEEP_ALIVE = "1d"
MAX_CONCURRENCY_PER_THREAD = 4
SAVE_LOCI_BATCH_WRITE_SIZE = int(os.getenv("SAVE_LOCI_BATCH_WRITE_SIZE", 500_000))
# Annotations are reported by filtered input row;
# this is very rapid, so to avoid spamming the user with messages,
# we only report every 5 million rows by default
ANNOTATION_MINIMUM_REPORTING_INTERVAL = int(
    os.getenv("ANNOTATION_MINIMUM_REPORTING_INTERVAL", 5_000_000)
)

# How many scroll requests for each worker to handle
PARALLEL_SCROLL_CHUNK_INCREMENT = 2
# Percentage of fetched records to report progress after
REPORTING_INTERVAL = 0.01
MINIMUM_RECORDS_TO_ENABLE_REPORTING = 100_000
# These are the fields that are required to define a locus
# They are used to filter the dosage matrix
FIELDS_TO_QUERY = ["chrom", "pos", "inputRef", "alt"]

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

    async def process_query(self, query: dict) -> tuple[NDArray[np.int32], NDArray]:
        doc_ids: list[int] = []
        loci: list[str] = []
        search_after = None  # Initialize search_after for pagination

        # Ensure there is a sort parameter in the query
        if "sort" not in query.get("body", {}):
            query.setdefault("body", {}).update(
                {"sort": [{"_id": "asc"}]}  # Sorting by ID in ascending order
            )

        while True:
            if search_after:
                query["body"]["search_after"] = search_after

            resp = await self.client.search(**query)

            if not resp["hits"]["hits"]:
                break  # Exit the loop if no more documents are found

            for doc in resp["hits"]["hits"]:
                src = doc["fields"]
                doc_id = int(doc["_id"])
                locus = (
                    f"{upper_chr(src['chrom'][0])}:{src['pos'][0]}:{src['inputRef'][0]}:{src['alt'][0]}"
                )
                doc_ids.append(doc_id)
                loci.append(locus)

            # Update search_after to the sort value of the last document retrieved
            search_after = resp["hits"]["hits"][-1]["sort"]

        if not doc_ids:
            return np.array([], dtype=np.int32), np.array([], dtype=object)

        self.last_reported_count += len(doc_ids)

        if self.last_reported_count > self.REPORT_INCREMENT:
            self.reporter.increment_and_write_progress_message.remote(
                self.last_reported_count, "Fetched", "variants"
            )
            self.last_reported_count = 0

        all_doc_ids = np.array(doc_ids, dtype=np.int32)
        all_loci = np.array(loci, dtype=object)

        sorted_indices = np.argsort(all_doc_ids, kind="stable")
        all_doc_ids = all_doc_ids[sorted_indices]
        all_loci = all_loci[sorted_indices]

        return all_doc_ids, all_loci

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

    # TODO 2024-04-26 @akotlar now that we use search_after within process_query,
    # it appears we no logner need to worry about slop
    # however, for now we will keep it, just in case there are hidden consequences
    # such as inflating the number of slices required past OpenSearch comfort for
    # the index size
    # or inflating the query size to the maximum allowed, which stresses OpenSearch too much
    expected_query_size_with_loss = max_query_size * 0.85

    num_slices_required = math.ceil(n_docs / expected_query_size_with_loss)
    if num_slices_required > max_slices:
        num_slices_required = max_slices

    logger.info("Constructed query with %d slices for %d hits", num_slices_required, n_docs)

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


def count_in_ranges_numpy(numbers):
    if len(numbers) == 0:
        return "No data provided"

    max_value = numbers.max() + 1  # Ensure the highest value is included in the range
    min_bin_size = 100  # Minimum bin size
    max_bins = 100  # Maximum number of bins

    # Calculate optimal bin size
    optimal_bin_size = max(min_bin_size, np.ceil(max_value / max_bins))

    # Calculate the number of bins using the optimal bin size
    num_bins = int(np.ceil(max_value / optimal_bin_size))

    # Define the edges of the bins
    bins = np.linspace(0, num_bins * optimal_bin_size, num_bins + 1)

    # Create the histogram
    counts, edges = np.histogram(numbers, bins=bins)

    # Build a string with non-zero bins
    result = ""
    for i in range(len(counts)):
        if counts[i] > 0:
            result += f"{int(edges[i])}-{int(edges[i+1]-1)}: {counts[i]}\n"

    return result.strip()  # Remove the last newline character


def sort_loci_and_doc_ids(
    results: list[tuple[NDArray[np.int32], NDArray]]
) -> tuple[NDArray[np.int32], NDArray, int]:
    if len(results) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=object), 0

    process = psutil.Process(os.getpid())
    start = time.time()

    # Initialize lists to collect all document IDs and loci
    all_doc_ids = []
    all_loci = []

    # Collect results
    n_hits = 0
    n_hits_list: list[int] = []
    for doc_ids, loci in results:
        all_doc_ids.append(doc_ids)
        all_loci.append(loci)
        n_hits += len(doc_ids)
        n_hits_list.append(len(doc_ids))

    try:
        n_hits_list_np: NDArray[np.uint32] = np.array(n_hits_list, dtype=np.uint32)
        del n_hits_list
        logger.info("Query hits per slice distribution:\n%s", count_in_ranges_numpy(n_hits_list_np))
        del n_hits_list_np
    except Exception as e:
        logger.warning("Failed to calculate bins due to %s", e)

    logger.info("Memory usage after query collection loop: %s MB", process.memory_info().rss / 1024**2)

    # Concatenate all results into a single array
    all_doc_ids_np = np.concatenate(all_doc_ids)
    del all_doc_ids
    all_loci_np = np.concatenate(all_loci)
    del all_loci

    # Log the memory usage after concatenation
    logger.info(
        "Memory usage after concatenating query results: %s MB", process.memory_info().rss / 1024**2
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start
    logger.info("Time to process and concatenate: %s seconds", elapsed_time)

    start = time.time()
    # Get the indices that would sort the loci array
    sorted_indices = np.argsort(all_doc_ids_np, kind="stable")

    logger.info("Sorting indices took %s seconds", time.time() - start)
    logger.info(
        "Memory usage after sorting indices: %s (MB)",
        process.memory_info().rss / 1024**2,
    )

    start = time.time()
    # Perform in-place sorting by reassigning sorted values back into the original arrays
    all_doc_ids_np = all_doc_ids_np[sorted_indices]
    all_loci_np = all_loci_np[sorted_indices]

    logger.info("Sorting results using sorted indices took %s seconds", time.time() - start)
    logger.info(
        "Memory usage after sorting query results: %s (MB)",
        process.memory_info().rss / 1024**2,
    )

    return all_doc_ids_np, all_loci_np, n_hits


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

    current_target_index = 0
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
            header_written = False
            header_fields = None
            filters: list[Callable[[list[bytes]], bool]] = []
            start = time.time()
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
                        p.stdin.write(line)
                        stats_fh.stdin.write(line)

                        n_retained += 1
                        filtered_loci += loci_sorted[current_target_index] + "\n"

                        if current_target_index % SAVE_LOCI_BATCH_WRITE_SIZE == 0:
                            loci_fh.write(filtered_loci)
                            filtered_loci = ""

                    # Move to the next target line number, if any
                    current_target_index += 1

                    if current_target_index >= n_hits:
                        break

                if (i + 1) % reporting_interval == 0:
                    end = time.time()

                    reporter.message.remote(  # type: ignore
                        (
                            f"Annotation: Filtered {i + 1} variants. {n_retained} kept. "
                            f"Took {end - start:.0f} seconds."
                        )
                    )
                    start = time.time()

            if len(filtered_loci) > 0:
                loci_fh.write(filtered_loci)
                filtered_loci = ""

            loci_fh.close()
            p.stdin.close()  # Close the stdin to signal that we're done sending input
            stats_fh.stdin.close()  # Close the stdin to signal that we're done sending input

            p.wait()
            stats_fh.wait()

            reporter.message.remote("Annotation: Completed filtering.")  # type: ignore

    reporter.message.remote(f"Annotation: {n_retained} variants survived filtering.")  # type: ignore

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
        os.path.exists(parent_dosage_matrix_path)
        and os.stat(parent_dosage_matrix_path).st_size > 0
        and os.path.exists(loci_file_path)
        and os.stat(loci_file_path).st_size > 0
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
) -> SaveJobResults:
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
    del loci_sorted
    gc.collect()

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

    return SaveJobResults(
        output_file_names=outputs,
        total_annotated=n_results,
        total_skipped=n_hits - n_results,
    )


async def go(  # pylint:disable=invalid-name
    job_data: SaveJobData, search_conf: dict, publisher: ProgressPublisher, queue_config_path: str
) -> SaveJobResults:
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

    reporter.increment_and_write_progress_message.remote(  # type: ignore
        0, "Fetched", "variants", force=True
    )
    reporter.clear_progress.remote()  # type: ignore

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

    if n_hits != num_docs:
        raise RuntimeError(
            "Number of hits does not match the number of documents. Expected %d, got %d"
            % (num_docs, n_hits)
        )

    reporter.message.remote(  # type: ignore
        f"OK: The number of fetched variants ({n_hits}) equals the number expected ({num_docs})"
    )

    outputs = filter_annotation_and_dosage_matrix(
        job_data=job_data,
        reporter=reporter,
        doc_ids_sorted=doc_ids_sorted,
        loci_sorted=loci_sorted,
        n_hits=n_hits,
        queue_config_path=queue_config_path,
    )

    return outputs
