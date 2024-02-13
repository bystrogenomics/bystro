from msgspec import json

from bystro.search.utils.annotation import AnnotationOutputs, StatisticsOutputs
from bystro.search.utils.messages import (
    IndexJobData,
    IndexJobResults,
    IndexJobCompleteMessage,
    SaveJobData,
    SaveJobResults,
    SaveJobCompleteMessage,
)


def test_index_job_data_camel_decamel():
    job_data = IndexJobData(
        submission_id="foo",
        input_dir="input_dir",
        out_dir="out_dir",
        input_file_names=AnnotationOutputs(
            annotation="annotation",
            sample_list="sample_list",
            log="log",
            config="config",
            statistics=StatisticsOutputs(json="json", tab="tab", qc="qc"),
            dosage_matrix_out_path="dosage_matrix_out_path",
            header="header",
            archived=None,
        ),
        index_name="index_name",
        assembly="assembly",
        index_config_path="index_config_path",
        field_names=["field1", "field2"],
    )

    serialized_values = json.encode(job_data)
    expected_value = {
        "submissionId": "foo",
        "inputDir": "input_dir",
        "outDir": "out_dir",
        "inputFileNames": {
            "annotation": "annotation",
            "sampleList": "sample_list",
            "log": "log",
            "config": "config",
            "statistics": {
                "json": "json",
                "tab": "tab",
                "qc": "qc",
            },
            "dosageMatrixOutPath": "dosage_matrix_out_path",
            "header": "header",
            "archived": None,
        },
        "indexName": "index_name",
        "assembly": "assembly",
        "indexConfigPath": "index_config_path",
        "fieldNames": ["field1", "field2"],
    }
    serialized_expected_value = json.encode(expected_value)

    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=IndexJobData)
    assert deserialized_values == job_data

def test_index_job_results_camel_decamel():
    job_results = IndexJobResults(
        index_config_path="index_config_path",
        field_names=["field1", "field2"]
    )

    serialized_values = json.encode(job_results)
    expected_value = {
        "indexConfigPath": "index_config_path",
        "fieldNames": ["field1", "field2"]
    }
    serialized_expected_value = json.encode(expected_value)

    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=IndexJobResults)
    assert deserialized_values == job_results

def test_index_job_complete_message_camel_decamel():
    job_results = IndexJobResults(
        index_config_path="index_config_path",
        field_names=["field1", "field2"]
    )
    completed_msg = IndexJobCompleteMessage(
        submission_id="foo",
        results=job_results
    )

    serialized_values = json.encode(completed_msg)
    expected_value = {
        "submissionId": "foo",
        "event": "completed",
        "results": {
            "indexConfigPath": "index_config_path",
            "fieldNames": ["field1", "field2"]
        }
    }
    serialized_expected_value = json.encode(expected_value)

    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=IndexJobCompleteMessage)
    assert deserialized_values == completed_msg

def test_save_job_data_camel_decamel():
    job_data = SaveJobData(
        submission_id="submit1",
        assembly="assembly",
        query_body={"query": "body"},
        input_dir="input_dir",
        input_file_names=AnnotationOutputs(
            annotation="annotation",
            sample_list="sample_list",
            log="log",
            config="config",
            statistics=StatisticsOutputs(json="json", tab="tab", qc="qc"),
            dosage_matrix_out_path="dosage_matrix_out_path",
            header="header",
            archived=None,
        ),
        index_name="index_name",
        output_base_path="output_base_path",
        field_names=["field1", "field2"],
        pipeline=None,
    )

    serialized_values = json.encode(job_data)
    expected_value = {
        "submissionId": "submit1",
        "assembly": "assembly",
        "queryBody": {"query": "body"},
        "inputDir": "input_dir",
        "inputFileNames": {
            "annotation": "annotation",
            "sampleList": "sample_list",
            "log": "log",
            "config": "config",
            "statistics": {
                "json": "json",
                "tab": "tab",
                "qc": "qc",
            },
            "dosageMatrixOutPath": "dosage_matrix_out_path",
            "header": "header",
            "archived": None,
        },
        "indexName": "index_name",
        "outputBasePath": "output_base_path",
        "fieldNames": ["field1", "field2"],
        "pipeline": None,
    }
    serialized_expected_value = json.encode(expected_value)

    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=SaveJobData)
    assert deserialized_values == job_data

def test_save_job_results_camel_decamel():
    job_results = SaveJobResults(
        output_file_names=AnnotationOutputs(
            annotation="annotation",
            sample_list="sample_list",
            log="log",
            config="config",
            statistics=StatisticsOutputs(json="json", tab="tab", qc="qc"),
            dosage_matrix_out_path="dosage_matrix_out_path",
            header="header",
            archived=None,
        )
    )

    serialized_values = json.encode(job_results)
    expected_value = {
        "outputFileNames": {
            "annotation": "annotation",
            "sampleList": "sample_list",
            "log": "log",
            "config": "config",
            "statistics": {
                "json": "json",
                "tab": "tab",
                "qc": "qc",
            },
            "dosageMatrixOutPath": "dosage_matrix_out_path",
            "header": "header",
            "archived": None,
        }
    }
    serialized_expected_value = json.encode(expected_value)

    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=SaveJobResults)
    assert deserialized_values == job_results

def test_save_job_complete_message_camel_decamel():
    job_results = SaveJobResults(
        output_file_names=AnnotationOutputs(
            annotation="annotation",
            sample_list="sample_list",
            log="log",
            config="config",
            statistics=StatisticsOutputs(json="json", tab="tab", qc="qc"),
            dosage_matrix_out_path="dosage_matrix_out_path",
            header="header",
            archived=None,
        )
    )
    completed_msg = SaveJobCompleteMessage(
        submission_id="submit1",
        results=job_results
    )

    serialized_values = json.encode(completed_msg)
    expected_value = {
        "submissionId": "submit1",
        "event": "completed",
        "results": {
            "outputFileNames": {
                "annotation": "annotation",
                "sampleList": "sample_list",
                "log": "log",
                "config": "config",
                "statistics": {
                    "json": "json",
                    "tab": "tab",
                    "qc": "qc",
                },
                "dosageMatrixOutPath": "dosage_matrix_out_path",
                "header": "header",
                "archived": None,
            }
        }
    }
    serialized_expected_value = json.encode(expected_value)

    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=SaveJobCompleteMessage)
    assert deserialized_values == completed_msg