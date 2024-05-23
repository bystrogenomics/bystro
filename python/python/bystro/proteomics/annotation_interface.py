"""Query an annotation file and return a list of sample_ids and genes meeting the query criteria."""

import copy
import logging
import math
from typing import Any, Callable

import asyncio
from msgspec import Struct
import nest_asyncio  # type: ignore
import numpy as np

import pandas as pd
from opensearchpy import AsyncOpenSearch

from bystro.api.auth import CachedAuth
from bystro.api.search import get_async_proxied_opensearch_client
from bystro.proteomics.fragpipe_tandem_mass_tag import TandemMassTagDataset
from bystro.search.utils.opensearch import gather_opensearch_args


logger = logging.getLogger(__file__)

HETEROZYGOTE_DOSAGE = 1
HOMOZYGOTE_DOSAGE = 2
MISSING_GENO_DOSAGE = np.nan
ONE_DAY = "1d"  # default keep_alive time for opensearch point in time index

nest_asyncio.apply()

# Fields that may look numeric but are lexical
DEFAULT_NOT_NUMERIC_FIELDS = [
    "pos",
    "vcfPos",
    "clinvarVcf.RS",
    "id",
    "gnomad.exomes.id",
    "gnomad.genomes.id",
    "clinvarVcf.id",
    "refSeq.codonNumber",
    "homozygotes",
    "heterozygotes",
    "missingGenos",
    "clinvarVcf.ALLELEID",
    "clinvarVcf.DBVARID",
    "clinvarVcf.ORIGIN",
]

DEFAULT_PRIMARY_KEYS = {
    "gnomad.genomes": "id",
    "gnomad.exomes": "id",
    "clinvarVcf": "id",
    "dbSNP": "id",
    "nearestTss.refSeq": "name",
    "nearest.refSeq": "name",
    "refSeq": "name",
}

ALWAYS_INCLUDED_FIELDS = ["chrom", "pos", "vcfPos", "inputRef", "alt", "type", "id"]
LINK_GENERATED_COLUMN = "locus"

CHROM_FIELD = "chrom"
POS_FIELD = "pos"
INPUT_REF_FIELD = "inputRef"
ALT_FIELD = "alt"
TYPE_FIELD = "type"


def looks_like_float(val):
    try:
        val = float(val)
    except ValueError:
        return False

    return True


def looks_like_number(val):
    not_float = False
    not_int = False
    try:
        return True, float(val)
    except ValueError:
        not_float = True

    try:
        return True, int(val)
    except ValueError:
        not_int = True

    return not (not_float and not_int), val


def transform_fields_with_dynamic_arity(
    data_structure, alt_field, track, primary_keys: dict[str, str] | None = None
):
    if primary_keys is None:
        primary_keys = DEFAULT_PRIMARY_KEYS

    def calculate_number_of_positions():
        is_number, val = looks_like_number(alt_field)
        if is_number:
            return int(min(abs(val), 32))
        return 2 if len(alt_field) >= 2 else 1

    positions_count = calculate_number_of_positions()

    def calculate_max_arity_for_position(position_data):
        if isinstance(position_data, list):
            return len(position_data)

        arity_key = primary_keys.get(track)

        max_arity = 0
        if arity_key is not None:
            if arity_key not in position_data:
                # Return 1 because the primary key was not selected
                # which means that the key values are relative to is not present
                # so we cannot separate the values into multiple records based on the primary key
                # so the best we can do is flatten the array of output values
                max_arity = 1
            else:
                if not isinstance(position_data[arity_key], list):
                    raise RuntimeError(
                        f"Expected list for track {track}, key {arity_key}, found {position_data[arity_key]}"
                    )
                max_arity = len(position_data[arity_key])
        else:
            for field in position_data.values():
                max_arity = max(max_arity, len(field))

        return max_arity

    def transform_position_data(position_data, keys):
        max_arity = calculate_max_arity_for_position(position_data)
        position_result = []

        for i in range(max_arity):
            item_info = {}
            for key in keys:
                if max_arity == 1:
                    # Due to deduplication, where we output 1 value
                    # when all values are identical for a key
                    # it is possible to have max_arity 1 for a primary_key,
                    # but higher arity in other fields
                    value = position_data[key]
                else:
                    value = (
                        position_data[key][0] if len(position_data[key]) == 1 else position_data[key][i]
                    )

                if isinstance(value, list):
                    value = _flatten(value)

                    if len(value) == 1:
                        value = value[0]

                item_info[key] = value
            position_result.append(item_info)

        return position_result

    def transform_position_data_for_no_key_data(position_data):
        max_arity = calculate_max_arity_for_position(position_data)
        position_result = []

        for i in range(max_arity):
            value = position_data[0] if len(position_data) == 1 else position_data[i]
            position_result.append(value[0] if isinstance(value, list) and len(value) == 1 else value)

        return position_result

    keys = None
    if not isinstance(data_structure, list):
        keys = list(data_structure.keys())

    transformed_data = []

    for position_index in range(positions_count):
        if keys is None:
            transformed_data.append(
                transform_position_data_for_no_key_data(
                    data_structure[0] if len(data_structure) == 1 else data_structure[position_index]
                )
            )
            continue

        position_data = {
            key: (
                data_structure[key][0]
                if len(data_structure[key]) == 1
                else data_structure[key][position_index]
            )
            for key in keys
        }
        transformed_data.append(transform_position_data(position_data, keys))

    return transformed_data


def generate_desired_structs_of_arrays(document: dict[str, Any]):
    """
    Generate desired structures of arrays.

    Args:
        document: Document

    Returns:
        Desired structures of arrays
    """
    result = {}

    def transform_object(obj):
        return obj

    def traverse_and_transform(obj: dict[str, Any], current_path=""):
        has_nested_objects = False

        for key, value in obj.items():
            new_path = f"{current_path}.{key}" if current_path else key

            if isinstance(value, dict):
                has_nested_objects = True
                traverse_and_transform(value, new_path)

        if not has_nested_objects or current_path:
            result[current_path] = transform_object(obj)

    traverse_and_transform(document)

    all_keys = document.keys()
    for key in all_keys:
        if key not in result:
            result[key] = document[key]

    for key in list(result.keys()):
        nested_keys = [k for k in result if k.startswith(f"{key}.")]
        for nested_key in nested_keys:
            sub_key = nested_key[len(key) + 1 :]
            if sub_key in result[key]:
                del result[key][sub_key]

    return result


def sort_keys(result: dict[str, Any], drop_alt=False) -> list[str]:
    """
    Sort keys in a dictionary.

    Args:
        result: Result dictionary
        drop_alt: Drop the "alt" key

    Returns:
        Sorted keys
    """
    keys = list(result.keys())

    if drop_alt and "alt" in keys:
        keys.remove("alt")

    keys.sort(key=str.lower)

    if "id" in keys:
        keys.remove("id")
        keys.insert(0, "id")

    return keys


def track_of_objects_to_track_of_arrays(
    data, track_name="", not_numeric_fields: list[str] | None = None
) -> list | int | float | str | None:
    """
    Convert a track of objects to a track of arrays.

    Args:
        data: Track of objects
        track_name: Track name
        not_numeric_fields: Fields that are not numeric

    Returns:
        Track of arrays
    """
    if not_numeric_fields is None:
        not_numeric_fields = DEFAULT_NOT_NUMERIC_FIELDS

    def convert_and_sort(obj, convert_key=""):
        if obj is None:
            return None

        if not isinstance(obj, (dict, list)):
            if convert_key not in not_numeric_fields:
                num = obj

                if convert_key not in not_numeric_fields and looks_like_float(obj):
                    num = float(obj)
                if not isinstance(num, (int, float)):
                    return obj
                if num == 0:
                    return 0
                abs_num = abs(num)
                if abs_num < 0.0001 or abs_num > 1000000:
                    return f"{num:.4e}"
                return round(num, 4)
            return obj

        if isinstance(obj, list):
            return [convert_and_sort(element, convert_key) for element in obj]

        keys = sort_keys(obj, True)

        if not keys:
            return []

        if len(keys) == 1:
            return [
                keys[0],
                convert_and_sort(obj[keys[0]], f"{convert_key}.{keys[0]}" if convert_key else keys[0]),
            ]

        return [
            [key, convert_and_sort(obj[key], f"{convert_key}.{key}" if convert_key else key)]
            for key in keys
        ]

    return convert_and_sort(data, track_name)


def fill_query_results_object(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Fill the query results object with the desired structure.

    Args:
        hits: List of hits from OpenSearch query

    Returns:
        List of query results objects
    """
    query_results_unique = []

    for row_result_obj in hits:
        row = {}

        flattened_data = generate_desired_structs_of_arrays(row_result_obj["_source"])

        for track_name, value in flattened_data.items():
            if not value:
                continue
            row[track_name] = transform_fields_with_dynamic_arity(
                value, row_result_obj["_source"]["alt"][0][0][0], track_name
            )

        query_results_unique.append(row)

    return query_results_unique


def query_results_to_array_of_structs(results_obj: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert query results to a struct of arrays.

    Args:
        results_obj: Query results object

    Returns:
        Struct of arrays
    """
    return [
        {
            track_name: track_of_objects_to_track_of_arrays(row[track_name], track_name)
            for track_name in row
        }
        for row in results_obj
    ]


def flatten_2d_array(d):
    """
    Flatten 2D arrays in a dictionary.

    Args:
        d: Dictionary

    Returns:
        Flattened dictionary
    """

    def is_2d_array(value):
        if isinstance(value, list) and all(isinstance(i, list) for i in value):
            return True
        return False

    def recurse(d):
        for key, value in d.items():
            if is_2d_array(value):
                d[key] = _flatten(value)
            elif isinstance(value, dict):
                recurse(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        recurse(item)

    recurse(d)
    return d


def process_dict_based_on_pos_length(d: dict[str, Any]):
    """
    Process a dictionary based on the length of the "pos" field.

    Args:
        d: Dictionary

    Returns:
        Processed dictionary
    """
    # Process 2D arrays to join or convert to single values
    flatten_2d_array(d)

    # Determine the length of the "pos" field
    pos_length = len(d["pos"])

    if pos_length == 1:
        # Convert all array values to scalars
        for key in d:
            if isinstance(d[key], list):
                d[key] = d[key][0]
        d[LINK_GENERATED_COLUMN] = (
            f"{d[CHROM_FIELD]}:{d[POS_FIELD]}:{d[INPUT_REF_FIELD]}:{d[ALT_FIELD]}:{d[TYPE_FIELD]}"
        )
        return [d]

    # Create an array of dictionaries, each corresponding to one index
    result = []
    for i in range(pos_length):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, list) and i < len(value):
                if isinstance(value[i], list) and len(value[i]) == 1:
                    new_dict[key] = value[i][0]
                else:
                    new_dict[key] = value[i]
            else:
                new_dict[key] = value

        new_dict[LINK_GENERATED_COLUMN] = (
            f"{new_dict[CHROM_FIELD]}:{new_dict[POS_FIELD]}:{new_dict[INPUT_REF_FIELD]}:{new_dict[ALT_FIELD]}:{new_dict[TYPE_FIELD]}"
        )
        result.append(new_dict)

    return result


def flatten_nested_dicts(dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Flatten nested dictionaries.

    Args:
        dicts: List of dictionaries

    Returns:
        Flattened list of dictionaries
    """
    flattened_dicts = []

    def flatten_dict(d, parent_key="", sep="."):
        items = []

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    for d in dicts:
        flattened_dicts.append(flatten_dict(d))

    return flattened_dicts


class OpenSearchQueryConfig(Struct):
    """Represent parameters for configuring OpenSearch queries."""

    max_query_size: int = 10_000
    max_slices: int = 1024
    keep_alive: str = ONE_DAY


OPENSEARCH_QUERY_CONFIG = OpenSearchQueryConfig()


def _flatten(xs: Any) -> list[Any]:  # noqa: ANN401 (`Any` is really correct here)
    """Flatten an arbitrarily nested list."""
    if not isinstance(xs, list):
        return [xs]
    return sum([_flatten(x) for x in xs], [])


def _get_nested_field(data, field_path):
    """Recursively fetch nested field values using dot notation."""
    keys = field_path.split(".")
    value = data
    for key in keys:
        try:
            value = value[key]
        except (KeyError, TypeError):
            return None  # Returns None if the field path is not found
    return value


def _get_samples_genes_dosages_from_hit(
    hit: dict[str, Any], fields: list[str] | None = None
) -> pd.DataFrame:
    """Given a document hit, return a dataframe of samples,
    genes and dosages with specified additional fields.
    """

    source = hit["_source"]
    # Base required fields
    chrom = _flatten(source["chrom"])[0]

    pos = _flatten(source["pos"])[0]
    _id = _flatten(source["id"])[0]
    vcf_pos = _flatten(source["vcfPos"])[0]
    input_ref = _flatten(source.get("inputRef", [None]))[0]
    ref = _flatten(source["ref"])[0]
    alt = _flatten(source["alt"])[0]
    gene_names = _flatten(
        source["refSeq"]["name2"]
    )  # guaranteed to be unique or to belong to different transcripts
    transcript_names = _flatten(source["refSeq"]["name"])
    protein_names = _flatten(source["refSeq"].get("protAcc", [None]))
    is_canonical = [
        x == "true" or x is True if x else False for x in _flatten(source["refSeq"].get("isCanonical"))
    ]
    gnomad_genomes_af = _flatten(_get_nested_field(source, "gnomad.genomes.AF"))[0]
    gnomad_exomes_af = _flatten(_get_nested_field(source, "gnomad.exomes.AF"))[0]

    if len(is_canonical) != len(transcript_names):
        is_canonical = [is_canonical[0]] * len(transcript_names)

    if len(protein_names) != len(transcript_names):
        protein_names = [protein_names[0]] * len(transcript_names)

    if len(gene_names) != len(transcript_names):
        gene_names = [gene_names[0]] * len(transcript_names)

    heterozygotes = _flatten(source.get("heterozygotes", []))
    homozygotes = _flatten(source.get("homozygotes", []))
    missing_genos = _flatten(source.get("missingGenos", []))

    heterozygosity = _flatten(source["heterozygosity"])[0]
    homozygosity = _flatten(source["homozygosity"])[0]
    missingness = _flatten(source["missingness"])[0]

    fields_to_add = list(
        filter(
            lambda x: x
            not in [
                "chrom",
                "pos",
                "id",
                "vcfPos",
                "inputRef",
                "ref",
                "alt",
                "refSeq.name2",
                "refSeq.name",
                "refSeq.protAcc",
                "refSeq.isCanonical",
                "heterozygotes",
                "homozygotes",
                "missingGenos",
                "heterozygosity",
                "homozygosity",
                "missingness",
                "gnomad.genomes.AF",
                "gnomad.exomes.AF",
            ],
            fields if fields is not None else [],
        )
    )

    rows = []
    for gene_idx, gene_name in enumerate(gene_names):
        for sample_list, dosage_label in [
            (heterozygotes, 1),
            (homozygotes, 2),
            (missing_genos, -1),
        ]:
            for sample_id in sample_list:
                row = {
                    "sample_id": sample_id,
                    "chrom": chrom,
                    "vcf_pos": vcf_pos,
                    "pos": pos,
                    "id": _id,
                    "ref": ref,
                    "input_ref": input_ref,
                    "alt": alt,
                    "gene_name": gene_name,
                    "transcript_name": transcript_names[gene_idx],
                    "protein_name": protein_names[gene_idx],
                    "is_canonical": is_canonical[gene_idx],
                    "dosage": dosage_label,
                    "heterozygosity": heterozygosity,
                    "homozygosity": homozygosity,
                    "missingness": missingness,
                    "gnomad_genomes_af": gnomad_genomes_af,
                    "gnomad_exomes_af": gnomad_exomes_af,
                }
                # Add additional fields
                for field in fields_to_add:
                    row[field] = _get_nested_field(source, field)

                    if row[field] is not None:
                        row[field] = _flatten(row[field])

                        if len(row[field]) == 1:
                            row[field] = row[field][0]
                        else:
                            row[field] = tuple(row[field])
                rows.append(row)

    return pd.DataFrame(rows)


async def execute_query(
    client: AsyncOpenSearch, query: dict, fields: list[str] | None = None
) -> pd.DataFrame:
    f"""
    Execute an OpenSearch query and return the results as a DataFrame.

    Args:
        client: OpenSearch client
        query: OpenSearch query
        fields: Fields to include in the DataFrame. {ALWAYS_INCLUDED_FIELDS} will always be included

    Returns:
        DataFrame of query results
    """
    results: list[dict] = []
    search_after = None  # Initialize search_after for pagination

    # Ensure there is a sort parameter in the query
    if "sort" not in query.get("body", {}):
        query.setdefault("body", {}).update(
            {"sort": [{"_id": "asc"}]}  # Sorting by ID in ascending order
        )

    while True:
        if search_after:
            query["body"]["search_after"] = search_after

        resp = await client.search(**query)

        if not resp["hits"]["hits"]:
            break  # Exit the loop if no more documents are found

        results.extend(resp["hits"]["hits"])

        # Update search_after to the sort value of the last document retrieved
        search_after = resp["hits"]["hits"][-1]["sort"]

    return process_query_response(results, fields)


def process_query_response(hits: list[dict[str, Any]], fields: list[str] | None = None) -> pd.DataFrame:
    """Postprocess query response from opensearch client."""
    num_hits = len(hits)

    if num_hits == 0:
        return pd.DataFrame()

    results_obj = fill_query_results_object(hits)

    rows = []
    for row in results_obj:
        rows.extend(process_dict_based_on_pos_length(row))

    rows = flatten_nested_dicts(rows)

    # samples_genes_dosages_df = pd.concat(
    #     [_get_samples_genes_dosages_from_hit(hit, fields) for hit in hits]
    # )
    # we may have multiple variants per gene in the results, so we
    # need to drop duplicates here.
    if fields is not None:
        cols = ALWAYS_INCLUDED_FIELDS + [LINK_GENERATED_COLUMN]

        cols += [
            field for field in fields if field not in cols
        ]

        return pd.DataFrame(rows, columns=cols)
    return pd.DataFrame(rows)


async def async_get_num_slices(
    client: AsyncOpenSearch,
    index_name: str,
    query: dict[str, Any],
) -> tuple[int, int]:
    """Count number of hits for the index."""
    get_num_slices_query = query["body"].copy()
    get_num_slices_query.pop("sort", None)
    get_num_slices_query.pop("track_total_hits", None)

    response = await client.count(body=get_num_slices_query, index=index_name)

    n_docs: int = response["count"]
    if n_docs < 1:
        err_msg = (
            f"Expected at least one document in `response['count']`, got response: {response} instead."
        )
        raise RuntimeError(err_msg)

    num_slices_necessary = math.ceil(n_docs / OPENSEARCH_QUERY_CONFIG.max_query_size)
    num_slices_planned = min(num_slices_necessary, OPENSEARCH_QUERY_CONFIG.max_slices)
    return max(num_slices_planned, 1), n_docs


async def async_run_annotation_query(
    query: dict[str, Any],
    index_name: str,
    fields: list[str] | None = None,
    cluster_opensearch_config: dict[str, Any] | None = None,
    bystro_api_auth: CachedAuth | None = None,
    additional_client_args: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run an annotation query and return a DataFrame of results.

    Args:
        query: OpenSearch query
        index_name: Index name
        fields: Additional fields to include in the DataFrame
        cluster_opensearch_config: Cluster OpenSearch configuration
        bystro_api_auth: Bystro API authentication
        additional_client_args: Additional arguments for OpenSearch client

    Returns:
        DataFrame of query results
    """
    if cluster_opensearch_config is not None and bystro_api_auth is not None:
        raise ValueError(
            "Cannot provide both cluster_opensearch_config and bystro_api_auth. Select one."
        )

    if bystro_api_auth is not None:
        # If auth is provided, use the proxied client
        job_id = index_name.split("_")[0]
        client = get_async_proxied_opensearch_client(bystro_api_auth, job_id, additional_client_args)
    elif cluster_opensearch_config is not None:
        search_client_args = gather_opensearch_args(cluster_opensearch_config)
        client = AsyncOpenSearch(**search_client_args)
    else:
        raise ValueError("Must provide either cluster_opensearch_config or bystro_api_auth.")

    num_slices, _ = await async_get_num_slices(client, index_name, query)

    point_in_time = await client.create_point_in_time(  # type: ignore[attr-defined]
        index=index_name, params={"keep_alive": OPENSEARCH_QUERY_CONFIG.keep_alive}
    )
    try:  # make sure we clean up the PIT index properly no matter what happens in this block
        pit_id = point_in_time["pit_id"]

        query["body"]["pit"] = {"id": pit_id}
        query["body"]["size"] = OPENSEARCH_QUERY_CONFIG.max_query_size
        query_results = []
        for slice_id in range(num_slices):
            slice_query = copy.deepcopy(query)
            if num_slices > 1:
                # Slice queries require max > 1
                slice_query["body"]["slice"] = {"id": slice_id, "max": num_slices}

            query_result = execute_query(client, query=slice_query, fields=fields)
            query_results.append(query_result)

        res = await asyncio.gather(*query_results)
        return pd.concat(res)
    except Exception as e:
        err_msg = (
            f"Encountered exception: {e!r} while running opensearch_query, "
            "deleting PIT index and exiting.\n"
            f"query: {query}\n"
            f"client: {client}\n"
            f"opensearch_query_config: {OPENSEARCH_QUERY_CONFIG}\n"
        )
        logger.exception(err_msg, exc_info=e)
        raise RuntimeError(err_msg) from e
    finally:
        await client.delete_point_in_time(body={"pit_id": pit_id})  # type: ignore[attr-defined]
        await client.close()


async def async_get_annotation_result_from_query(
    query_string: str,
    index_name: str,
    fields: list[str] | None = None,
    cluster_opensearch_config: dict[str, Any] | None = None,
    bystro_api_auth: CachedAuth | None = None,
    additional_client_args: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Given a query and index, return a dataframe of variant / sample_id records matching query."""

    if cluster_opensearch_config is not None and bystro_api_auth is not None:
        raise ValueError(
            "Cannot provide both cluster_opensearch_config and bystro_api_auth. Select one."
        )

    query = _build_opensearch_query_from_query_string(query_string, fields=fields)

    return await async_run_annotation_query(
        query,
        index_name,
        fields=fields,
        cluster_opensearch_config=cluster_opensearch_config,
        bystro_api_auth=bystro_api_auth,
        additional_client_args=additional_client_args,
    )


def get_annotation_result_from_query(
    query_string: str,
    index_name: str,
    fields: list[str] | None = None,
    cluster_opensearch_config: dict[str, Any] | None = None,
    bystro_api_auth: CachedAuth | None = None,
    additional_client_args: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Given a query and index, return a dataframe of variant / sample_id records matching query."""
    loop = asyncio.get_event_loop()
    coroutine = async_get_annotation_result_from_query(
        query_string,
        index_name,
        fields=fields,
        cluster_opensearch_config=cluster_opensearch_config,
        bystro_api_auth=bystro_api_auth,
        additional_client_args=additional_client_args,
    )

    return loop.run_until_complete(coroutine)


def _build_opensearch_query_from_query_string(
    query_string: str, fields: list[str] | None = None
) -> dict[str, Any]:
    base_query: dict[str, Any] = {
        "body": {
            "query": {
                "bool": {
                    "filter": {
                        "query_string": {
                            "default_operator": "AND",
                            "query": query_string,
                            "lenient": True,
                            "phrase_slop": 5,
                            "tie_breaker": 0.3,
                        },
                    },
                },
            },
            "sort": "_doc",
        }
    }

    all_fields = ALWAYS_INCLUDED_FIELDS.copy()
    for field in fields or []:
        if field not in all_fields:
            all_fields.append(field)

    if fields is not None:
        base_query["_source_includes"] = all_fields

    return base_query


def join_annotation_result_to_proteomics_dataset(
    query_result_df: pd.DataFrame,
    tmt_dataset: TandemMassTagDataset,
    get_tracking_id_from_genomic_sample_id: Callable[[str], str] = (lambda x: x),
    get_tracking_id_from_proteomic_sample_id: Callable[[str], str] = (lambda x: x),
) -> pd.DataFrame:
    """
    Args:
      query_result_df: pd.DataFrame containing result from get_annotation_result_from_query
      tmt_dataset: TamdemMassTagDataset
      get_tracking_id_from_proteomic_sample_id: Callable mapping proteomic sample IDs to tracking IDs
      get_tracking_id_from_genomic_sample_id: Callable mapping genomic sample IDs to tracking IDs
    """
    query_result_df = query_result_df.copy()
    proteomics_df = tmt_dataset.get_melted_abundance_df()

    query_result_df.sample_id = query_result_df.sample_id.apply(get_tracking_id_from_genomic_sample_id)
    proteomics_df.sample_id = proteomics_df.sample_id.apply(get_tracking_id_from_proteomic_sample_id)

    joined_df = query_result_df.merge(
        proteomics_df,
        left_on=["sample_id", "gene_name"],
        right_on=["sample_id", "gene_name"],
    )
    return joined_df
