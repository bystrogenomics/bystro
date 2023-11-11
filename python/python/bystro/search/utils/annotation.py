import os
from glob import glob
import logging
from os import path
import shutil
from typing import Optional, Any

from msgspec import Struct

logger = logging.getLogger(__name__)


class FileProcessorsConfig(Struct, frozen=True, forbid_unknown_fields=True):
    args: str
    program: str


class StatisticsOutputExtensions(Struct, frozen=True, forbid_unknown_fields=True):
    json: str = "statistics.json"
    tsv: str = "statistics.tsv"
    qc: str = "statistics.qc.tsv"


class StatisticsConfig(Struct, frozen=True, forbid_unknown_fields=True):
    dbSNPnameField: str = "dbSNP.name"
    siteTypeField: str = "refSeq.siteType"
    exonicAlleleFunctionField: str = "refSeq.exonicAlleleFunction"
    refField: str = "ref"
    homozygotesField: str = "homozygotes"
    heterozygotesField: str = "heterozygotes"
    altField: str = "alt"
    programPath: str = "bystro-stats"
    outputExtensions: StatisticsOutputExtensions = StatisticsOutputExtensions()

    @staticmethod
    def from_dict(annotation_config: dict[str, Any]):
        """Get statistics config from a dictionary"""
        stats_config: Optional[dict[str, Any]] = annotation_config.get("statistics")

        if stats_config is None:
            logger.warning(
                "No 'statistics' config found in supplied annotation config, using defaults"
            )
            return StatisticsConfig()

        if "outputExtensions" in stats_config:
            stats_config["outputExtensions"] = StatisticsOutputExtensions(
                **stats_config["outputExtensions"]
            )

        return StatisticsConfig(**stats_config)


class StatisticsOutputs(Struct, frozen=True, forbid_unknown_fields=True):
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


class AnnotationOutputs(Struct, frozen=True, forbid_unknown_fields=True):
    """
    Paths to all possible Bystro annotation outputs

    Attributes:
        output_dir: str
            Output directory
        archived: str
            Basename of the archive
        annotation: str
            Basename of the annotation TSV file, found inside the archive only
        sampleList: Optional[str]
            Basename of the sample list file, in the archive and output directory
        log: str
            Basename of the log file, in the archive and output directory
        statistics: StatisticsOutputs
            Basenames of the statistics files, in the archive and output directory
        header: Optional[str]
            Basename of the header file, in the archive and output directory
    """

    archived: str
    annotation: str
    sampleList: str
    log: str
    statistics: StatisticsOutputs
    header: str | None = None

    @staticmethod
    def from_path(
        output_dir: str,
        basename: str,
        compress: bool,
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

        stats = Statistics(output_base_path=os.path.join(output_dir, basename))
        statistics_tarball_members = StatisticsOutputs(
            json=f"{os.path.basename(stats.json_output_path)}",
            tab=f"{os.path.basename(stats.tsv_output_path)}",
            qc=f"{os.path.basename(stats.qc_output_path)}",
        )

        return (
            AnnotationOutputs(
                annotation=annotation,
                sampleList=sampleList,
                statistics=statistics_tarball_members,
                archived=archived,
                log=log,
            ),
            stats,
        )


class DelimitersConfig(Struct, frozen=True, forbid_unknown_fields=True):
    field: str = "\t"
    position: str = "|"
    overlap: str = chr(31)
    value: str = ";"
    empty_field: str = "NA"

    @staticmethod
    def from_dict(annotation_config: dict[str, Any]):
        """Get delimiters from a dictionary"""
        delim_config: Optional[dict[str, str]] = annotation_config.get("delimiters")

        if delim_config is None:
            logger.warning(
                "No 'delimiters' key found in supplied annotation config, using defaults"
            )
            return DelimitersConfig()

        return DelimitersConfig(**delim_config)


def get_config_file_path(
    config_path_base_dir: str, assembly: str, suffix: str = ".y*ml"
):
    """Get config file path"""
    paths = glob(path.join(config_path_base_dir, assembly + suffix))

    if not paths:
        raise ValueError(
            f"\n\nNo config path found for the assembly {assembly}. Exiting\n\n"
        )

    if len(paths) > 1:
        print("\n\nMore than 1 config path found, choosing first")

    return paths[0]


class Statistics:
    def __init__(
        self, output_base_path: str, annotation_config: dict[str, Any] | None = None
    ):
        if annotation_config is None:
            self._config = StatisticsConfig()
            self._delimiters = DelimitersConfig()
        else:
            self._config = StatisticsConfig.from_dict(annotation_config)
            self._delimiters = DelimitersConfig.from_dict(annotation_config)

        program_path = shutil.which(self._config.programPath)
        if not program_path:
            raise ValueError(
                f"Couldn't find statistics program {self._config.programPath}"
            )

        self.program_path = program_path
        self.json_output_path = (
            f"{output_base_path}.{self._config.outputExtensions.json}"
        )
        self.tsv_output_path = f"{output_base_path}.{self._config.outputExtensions.tsv}"
        self.qc_output_path = f"{output_base_path}.{self._config.outputExtensions.qc}"

    @property
    def stdin_cli_stats_command(self) -> str:
        value_delim = self._delimiters.value
        field_delim = self._delimiters.field
        empty_field = self._delimiters.empty_field

        het_field = self._config.heterozygotesField
        hom_field = self._config.homozygotesField
        site_type_field = self._config.siteTypeField
        ea_fun_field = self._config.exonicAlleleFunctionField
        ref_field = self._config.refField
        alt_field = self._config.altField
        dbSNP_field = self._config.dbSNPnameField

        statsProg = self.program_path

        dbSNPpart = f"-dbSnpNameColumn {dbSNP_field}" if dbSNP_field else ""

        return (
            f"{statsProg} -outJsonPath {self.json_output_path} -outTabPath {self.tsv_output_path} "
            f"-outQcTabPath {self.qc_output_path} -refColumn {ref_field} "
            f"-altColumn {alt_field} -homozygotesColumn {hom_field} "
            f"-heterozygotesColumn {het_field} -siteTypeColumn {site_type_field} "
            f"{dbSNPpart} -emptyField '{empty_field}' "
            f"-exonicAlleleFunctionColumn {ea_fun_field} "
            f"-primaryDelimiter '{value_delim}' -fieldSeparator '{field_delim}'"
        )
