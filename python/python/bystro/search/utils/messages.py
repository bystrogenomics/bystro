from bystro.beanstalkd.messages import (
    BaseMessage,
    CompletedJobMessage,
    Struct,
)
from bystro.search.utils.annotation import AnnotationOutputs
from bystro.search.save.hwe import HWEFilter
from bystro.search.save.binomial_maf import BinomialMafFilter


class IndexJobData(BaseMessage, frozen=True, forbid_unknown_fields=True, kw_only=True, rename="camel"):
    """Data for Indexing jobs received from beanstalkd"""

    input_dir: str
    out_dir: str
    input_file_names: AnnotationOutputs
    index_name: str
    assembly: str
    index_config_path: str | None = None
    field_names: list[str] | None = None


class IndexJobResults(Struct, frozen=True, forbid_unknown_fields=True, rename="camel"):
    index_config_path: str
    field_names: list[str]


class IndexJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True):
    results: IndexJobResults


PipelineType = list[BinomialMafFilter | HWEFilter] | None


class SaveJobData(BaseMessage, frozen=True, forbid_unknown_fields=True, kw_only=True, rename="camel"):
    """Data for SaveFromQuery jobs received from beanstalkd"""

    assembly: str
    query_body: dict
    input_dir: str
    input_file_names: AnnotationOutputs
    index_name: str
    output_base_path: str
    field_names: list[str]
    pipeline: PipelineType = None


class SaveJobResults(Struct, frozen=True, rename="camel"):
    output_file_names: AnnotationOutputs


class SaveJobCompleteMessage(CompletedJobMessage, frozen=True, kw_only=True, rename="camel"):
    results: SaveJobResults
