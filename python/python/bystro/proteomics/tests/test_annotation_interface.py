import pytest

from bystro.beanstalkd.worker import ProgressPublisher
from bystro.proteomics.annotation_interface import (
    OPENSEARCH_CONFIG,
    #    _make_output_string,
    _process_response,
    get_samples_and_genes,
    run_query_and_write_output_pure,
)
from bystro.proteomics.tests.test_response import TEST_RESPONSE
from bystro.search.utils.annotation import get_delimiters
from bystro.search.utils.messages import SaveJobData

SAMPLE_INDEX_NAME = "64c889415acb6d3b3e40e07b_63ddc9ce1e740e0020c39928"
DELIMITERS = get_delimiters()


def my_test_end_to_end2(query, field_names):
    query_copy = query.copy()
    print("example query:", query)
    fields = query_copy["query"]["bool"]["filter"]["query_string"].get("fields", [])
    print("fields:", fields)
    num_fields = len(fields) if fields is not None else 0
    print("num_fields:", num_fields)
    job_data = SaveJobData(
        submissionID="1337",
        assembly="hg38",
        queryBody=query_copy,
        indexName=SAMPLE_INDEX_NAME,
        outputBasePath="foo/bar",
        fieldNames=["discordant", *field_names],
    )
    publisher = ProgressPublisher(host="127.0.0.1", port=1337, queue="proteomics", message=None)
    run_query_and_write_output(job_data, SEARCH_CONF, publisher)


def my_test_end_to_end_pure(query, field_names):
    query_copy = query.copy()
    print("example query:", query)
    fields = query_copy["query"]["bool"]["filter"]["query_string"].get("fields", [])
    print("fields:", fields)
    num_fields = len(fields) if fields is not None else 0
    print("num_fields:", num_fields)
    job_data = SaveJobData(
        submissionID="1337",
        assembly="hg38",
        queryBody=query_copy,
        indexName=SAMPLE_INDEX_NAME,
        outputBasePath="foo/bar",
        fieldNames=["discordant", *field_names],
    )
    annotation_output = run_query_and_write_output_pure(job_data, SEARCH_CONF)
    return annotation_output


@pytest.mark.network
def test_get_samples_and_genes():
    user_query_string = "exonic (gnomad.genomes.af:<0.1 || gnomad.exomes.af:<0.1)"
    samples_and_genes_df = get_samples_and_genes(user_query_string, SAMPLE_INDEX_NAME)
    assert 1191 == len(samples_and_genes_df)


import copy

import numpy as np


def _make_output_string_reference_implementation(  # noqa: C901  type: ignore
    rows: list[list[list[list[str | None]]]], delims: dict[str, str]
) -> bytes:
    empty_field_char = delims["empty_field"]
    for row_idx, row in enumerate(rows):  # pylint:disable=too-many-nested-blocks
        # Some fields may just be missing; we won't store even the alt/pos [[]] structure for those
        for i, column in enumerate(row):
            if column is None:
                row[i] = empty_field_char
                continue

            # For now, we don't store multiallelics; top level array is placeholder only
            # With breadth 1
            if not isinstance(column, list):
                row[i] = str(column)
                continue

            for j, position_data in enumerate(column):
                if position_data is None:
                    column[j] = empty_field_char
                    continue

                if isinstance(position_data, list):
                    inner_values = []
                    for sub in position_data:
                        if sub is None:
                            inner_values.append(empty_field_char)
                            continue

                        if isinstance(sub, list):
                            inner_values.append(
                                delims["value"].join(
                                    str(x) if x is not None else empty_field_char for x in sub
                                )
                            )
                        else:
                            inner_values.append(str(sub))

                    column[j] = delims["position"].join(inner_values)

            row[i] = delims["overlap"].join(column)

        rows[row_idx] = delims["field"].join(row)

    return bytes("\n".join(rows) + "\n", encoding="utf-8")


import time


def tests__process_response():
    ans = _process_response(TEST_RESPONSE)
    assert len(ans) == 1191


# def test__make_output_string() -> None:
#     rows = [
#         np.array(
#             [
#                 [["0"]],
#                 None,
#                 [["1805"]],
#                 [[["MAPK7"], ["MAPK7"], ["MAPK7"], ["MAPK7"]]],
#             ],
#             dtype=object,
#         ),
#         np.array([[["0"]], None, [["1805"]], [["ABCA6"]]], dtype=object),
#         np.array([[["0"]], None, [["1847"]], [["APOBEC3F"]]], dtype=object),
#         np.array(
#             [
#                 [["0"]],
#                 None,
#                 [[["1847"], ["4805"]]],
#                 [[["PDE11A"], ["PDE11A"], ["PDE11A"], ["PDE11A"]]],
#             ],
#             dtype=object,
#         ),
#         np.array(
#             [[["0"]], None, [[["1805"], ["4805"]]], [["PCGF1"]]],
#             dtype=object,
#         ),
#         np.array(
#             [[["0"]], None, [[["1847"], ["4805"]]], [["TEX37"]]],
#             dtype=object,
#         ),
#         np.array(
#             [
#                 [["0"]],
#                 None,
#                 [["1847"]],
#                 [[["PAX3"], ["PAX3"], ["PAX3"], ["PAX3"], ["PAX3"], ["PAX3"]]],
#             ],
#             dtype=object,
#         ),
#         np.array(
#             [
#                 [["0"]],
#                 None,
#                 [["1847"]],
#                 [[["COL6A3"], ["COL6A3"], ["COL6A3"], ["COL6A3"], ["COL6A3"]]],
#             ],
#             dtype=object,
#         ),
#         np.array(
#             [[["0"]], None, [[["1847"], ["4805"]]], [["THSD4"]]],
#             dtype=object,
#         ),
#         np.array(
#             [[["0"]], None, [[["1805"], ["4805"]]], [["DNAH17"]]],
#             dtype=object,
#         ),
#         np.array(
#             [[["0"]], None, [["1805"]], [[["CNP"], ["CNP"]]]],
#             dtype=object,
#         ),
#         np.array([[["0"]], None, [["1805"]], [["HEATR5B"]]], dtype=object),
#         np.array([[["0"]], None, [["1805"]], [["PTCD3"]]], dtype=object),
#         np.array([[["0"]], None, [["1847"]], [["LCT"]]], dtype=object),
#         np.array(
#             [
#                 [["0"]],
#                 None,
#                 [["1847"]],
#                 [
#                     [
#                         ["COLEC11"],
#                         ["COLEC11"],
#                         ["COLEC11"],
#                         ["COLEC11"],
#                         ["COLEC11"],
#                         ["COLEC11"],
#                         ["COLEC11"],
#                         ["COLEC11"],
#                         ["COLEC11"],
#                         ["COLEC11"],
#                         ["COLEC11"],
#                     ]
#                 ],
#             ],
#             dtype=object,
#         ),
#         np.array([[["0"]], None, [["1847"]], [["ABCB11"]]], dtype=object),
#         np.array([[["0"]], None, [["1805"]], [["SF3B1"]]], dtype=object),
#         np.array(
#             [
#                 [["0"]],
#                 None,
#                 [["1847"]],
#                 [[["EPB41L5"], ["EPB41L5"], ["EPB41L5"]]],
#             ],
#             dtype=object,
#         ),
#         np.array(
#             [
#                 [["0"]],
#                 None,
#                 [["1805"]],
#                 [[["FBXO7"], ["FBXO7"], ["FBXO7"]]],
#             ],
#             dtype=object,
#         ),
#         np.array([[["0"]], None, [["1805"]], [["ST6GALNAC2"]]], dtype=object),
#         np.array(
#             [
#                 [["0"]],
#                 None,
#                 [[["1805"], ["4805"]]],
#                 [[["GAA"], ["GAA"], ["GAA"]]],
#             ],
#             dtype=object,
#         ),
#         np.array(
#             [
#                 [["0"]],
#                 None,
#                 [[["1847"], ["4805"]]],
#                 [[["ALMS1"], ["ALMS1"]]],
#             ],
#             dtype=object,
#         ),
#         np.array(
#             [[["0"]], None, [[["1805"], ["4805"]]], [["ALK"]]],
#             dtype=object,
#         ),
#         np.array([[["0"]], None, [["1847"]], [["BMPR2"]]], dtype=object),
#         np.array(
#             [[["0"]], None, [["1847"]], [[["SMC6"], ["SMC6"]]]],
#             dtype=object,
#         ),
#     ]
#     safe_rows = copy.deepcopy(rows)
#     safe_rows2 = copy.deepcopy(rows)

#     tic = time.time()
#     expected = _make_output_string_reference_implementation(safe_rows, delims=DELIMITERS)
#     toc = time.time()
#     ref_time = toc - tic
#     print("ref implementation:", ref_time)

#     tic = time.time()
#     actual = _make_output_string(safe_rows2, delims=DELIMITERS)
#     toc = time.time()
#     spec_time = toc - tic
#     print("spec implementation:", spec_time)
#     spec_ref_ratio = spec_time / ref_time
#     print("ratio of spec to ref walltime:", round(spec_ref_ratio, 2))
#     assert spec_ref_ratio < 2
#     assert expected == actual
