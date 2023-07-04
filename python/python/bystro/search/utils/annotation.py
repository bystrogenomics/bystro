import os
from glob import glob
from os import path

from msgspec import Struct


class StatisticsOutputs(Struct, frozen=True):
    """
        Paths to all possible Bystro statistics outputs

        Attributes:
            json: str
                Basename of the JSON statistics file
            tab: str
                Basename of the TSV statistics file
            qc: str
                Basename of the QC statistics file
    """
    json: str
    tab: str
    qc: str


class AnnotationOutputs(Struct, frozen=True):
    """
        Paths to all possible Bystro annotation outputs
        
        Attributes:
            output_dir: str
                Output directory
            archived: str
                Basename of the archive
            annotation: str
                Basename of the annotation TSV file, inside the archive
            sampleList: Optional[str]
                Basename of the sample list file, inside the archive
            log: Basename of the log file, inside the archive
            statistics: Optional[StatisticsOutputs]
                Basenames of the statistics files, inside the archive
    """
    output_dir: str
    archived: str
    annotation: str
    sampleList: str
    log: str
    statistics: StatisticsOutputs | None = None

    @staticmethod
    def from_path(
        output_dir: str,
        basename: str,
        compress: bool,
        generate_statistics: bool = True,
        make_dir: bool = True,
        make_dir_mode: int = 511,
    ):
        """Make AnnotationOutputs based on the output base path and the output options"""
        if make_dir:
            os.makedirs(output_dir, mode=make_dir_mode, exist_ok=True)

        if not os.path.isdir(output_dir):
            raise IOError(f"Output directory {output_dir} does not exist")

        log = f"{basename}.log"
        annotation = f"{basename}.annotation.tsv"

        if compress:
            annotation += ".gz"

        sampleList = f"{basename}.sample_list"

        archived = f"{basename}.tar"

        statistics = None
        if generate_statistics:
            statistics = StatisticsOutputs(
                json=f"{basename}.statistics.json",
                tab=f"{basename}.statistics.tsv",
                qc=f"{basename}.statistics.qc.tsv",
            )

        return AnnotationOutputs(
            output_dir=output_dir,
            annotation=annotation,
            sampleList=sampleList,
            statistics=statistics,
            archived=archived,
            log=log,
        )


_default_delimiters = {
    "field": "\t",
    "allele": "/",
    "position": "|",
    "overlap": chr(31),
    "value": ";",
    "empty_field": "!",
}


def get_delimiters(annotation_conf: dict | None = None):
    if annotation_conf:
        return annotation_conf.get("delimiters", _default_delimiters)
    return _default_delimiters


def get_config_file_path(config_path_base_dir: str, assembly: str, suffix: str = ".y*ml"):
    """Get config file path"""
    paths = glob(path.join(config_path_base_dir, assembly + suffix))

    if not paths:
        raise ValueError(f"\n\nNo config path found for the assembly {assembly}. Exiting\n\n")

    if len(paths) > 1:
        print("\n\nMore than 1 config path found, choosing first")

    return paths[0]
