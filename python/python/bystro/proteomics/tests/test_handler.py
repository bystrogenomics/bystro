from bystro.search.utils.messages import SaveJobData
from ruamel.yaml import YAML
from bystro.beanstalkd.worker import ProgressPublisher
from pathlib import Path
from bystro.proteomics.handler import run_query_and_write_output
from bystro.search.utils.annotation import AnnotationOutputs, StatisticsOutputs

with open(Path.home() / "bystro/config/opensearch.yml") as search_config_file:
    search_conf = YAML(typ="safe").load(search_config_file)
    search_conf["connection"]["request_timeout"] = 5


INDEX_NAME = "64c889415acb6d3b3e40e07b_63ddc9ce1e740e0020c39928"


def test_end_to_end():
    job_data = SaveJobData(
        submissionID="1337",
        assembly="hg38",
        queryBody={},
        indexName=INDEX_NAME,
        outputBasePath="foo/bar",
        fieldNames=["discordant"],
    )
    publisher = ProgressPublisher(host="127.0.0.1", port=1337, queue="proteomics", message=None)
    actual_annotation_output = run_query_and_write_output(job_data, search_conf, publisher)
    expected_annotation_output = AnnotationOutputs(
        archived="bar.tar",
        annotation="bar.annotation.tsv.gz",
        sampleList="bar.sample_list",
        log="bar.log",
        statistics=StatisticsOutputs(
            json="bar.statistics.json", tab="bar.statistics.tsv", qc="bar.statistics.qc.tsv"
        ),
    )
    assert expected_annotation_output == actual_annotation_output
