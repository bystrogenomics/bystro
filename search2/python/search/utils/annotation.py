import os
from typing import Optional

from msgspec import Struct

class StatisticsOutputs(Struct, frozen=True):
        json: str
        tab: str
        qc: str

class AnnotationOutputs(Struct, frozen=True):
    """Annotation Outputs"""
    annotation: str
    sampleList: str
    log: str
    statistics: StatisticsOutputs | None = None
    archived: str  | None = None

    @staticmethod
    def new_from_base_path(output_base_path: str, generate_statistics: bool, compress: bool, archive: bool):
        """Make AnnotationOutputs based on the output base path and the output options"""

        basename = os.path.basename(output_base_path)
        log = f"{basename}.log"
        annotation = f"{basename}.annotation.tsv"

        if compress:
            annotation += '.gz'

        sampleList = f"{basename}.sample_list"

        archived = None
        if archive:
            archived = f"{basename}.tar"

        statistics = None
        if generate_statistics:
            statistics = StatisticsOutputs(
                json=f"{basename}.statistics.json",
                tab=f"{basename}.statistics.tsv",
                qc=f"{basename}.statistics.qc.tsv"
            )

        return AnnotationOutputs(
            annotation=annotation,
            sampleList=sampleList,
            statistics=statistics,
            log=log,
            archived=archived
        )

_default_delimiters = {
    "field": "\t",
    "allele": "/",
    "position": "|",
    'overlap': "\\",
    "value": ";",
    "empty_field": "!",
}

def get_delimiters(annotation_conf: Optional[dict] = None):
    if annotation_conf:
        return annotation_conf.get("delimiters", _default_delimiters)
    return _default_delimiters