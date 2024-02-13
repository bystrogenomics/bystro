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


class StatisticsConfig(Struct, frozen=True, forbid_unknown_fields=True, rename="camel"):
    dbsnp_name_field: str = "dbSNP.name"
    site_type_field: str = "refSeq.siteType"
    exonic_allele_function_field: str = "refSeq.exonicAlleleFunction"
    ref_field: str = "ref"
    homozygotes_field: str = "homozygotes"
    heterozygotes_field: str = "heterozygotes"
    alt_field: str = "alt"
    program_path: str = "bystro-stats"
    output_extension: StatisticsOutputExtensions = StatisticsOutputExtensions()

    @staticmethod
    def from_dict(annotation_config: dict[str, Any]):
        """Get statistics config from a dictionary"""
        stats_config: Optional[dict[str, Any]] = annotation_config.get("statistics")

        if stats_config is None:
            logger.warning(
                "No 'statistics' config found in supplied annotation config, using defaults"
            )
            return StatisticsConfig()

        if "output_extension" in stats_config:
            stats_config["output_extension"] = StatisticsOutputExtensions(
                **stats_config["output_extension"]
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


class AnnotationOutputs(Struct, frozen=True, forbid_unknown_fields=True, rename="camel"):
    """
    Paths to all possible Bystro annotation outputs

    Attributes:
        output_dir: str
            Output directory
        annotation: str
            Basename of the annotation TSV file, in the output directory
        sample_list: Optional[str]
            Basename of the sample list file, in the output directory
        log: str
            Basename of the log file, in the output directory
        config: str
            Basename of the config file, in the output directory
        statistics: StatisticsOutputs
            Basenames of the statistics files, in the output directory
        dosage_matrix_out_path: str
            Basename of the dosage matrix, in the output directory
        header: Optional[str]
            Basename of the header file, in the output directory
        archived: Optional[str]
            Basename of the archived annotation file, in the output directory
    """

    annotation: str
    sample_list: str
    log: str
    config: str
    statistics: StatisticsOutputs
    dosage_matrix_out_path: str
    header: str | None = None
    archived: str | None = None

    @staticmethod
    def from_path(
        output_dir: str,
        basename: str,
        annotation_config_path: str,
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

        sample_list = f"{basename}.sample_list"

        stats = Statistics(output_base_path=os.path.join(output_dir, basename))
        statistics_output_members = StatisticsOutputs(
            json=f"{os.path.basename(stats.json_output_path)}",
            tab=f"{os.path.basename(stats.tsv_output_path)}",
            qc=f"{os.path.basename(stats.qc_output_path)}",
        )

        dosage = f"{basename}.dosage.feather"

        return (
            AnnotationOutputs(
                annotation=annotation,
                sample_list=sample_list,
                statistics=statistics_output_members,
                config=annotation_config_path,
                dosage_matrix_out_path=dosage,
                log=log,
            ),
            stats,
        )


class DelimitersConfig(Struct, frozen=True, forbid_unknown_fields=True):
    field: str = "\t"
    position: str = "|"
    overlap: str = "/"
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

        program_path = shutil.which(self._config.program_path)
        if not program_path:
            raise ValueError(
                f"Couldn't find statistics program {self._config.program_path}"
            )

        self.program_path = program_path
        self.json_output_path = (
            f"{output_base_path}.{self._config.output_extension.json}"
        )
        self.tsv_output_path = f"{output_base_path}.{self._config.output_extension.tsv}"
        self.qc_output_path = f"{output_base_path}.{self._config.output_extension.qc}"

    @property
    def stdin_cli_stats_command(self) -> str:
        value_delim = self._delimiters.value
        field_delim = self._delimiters.field
        empty_field = self._delimiters.empty_field

        het_field = self._config.heterozygotes_field
        hom_field = self._config.homozygotes_field
        site_type_field = self._config.site_type_field
        ea_fun_field = self._config.exonic_allele_function_field
        ref_field = self._config.ref_field
        alt_field = self._config.alt_field
        dbsnp_field = self._config.dbsnp_name_field

        prog = self.program_path

        dbsnp_part = f"-dbSnpNameColumn {dbsnp_field}" if dbsnp_field else ""

        return (
            f"{prog} -outJsonPath {self.json_output_path} -outTabPath {self.tsv_output_path} "
            f"-outQcTabPath {self.qc_output_path} -refColumn {ref_field} "
            f"-altColumn {alt_field} -homozygotesColumn {hom_field} "
            f"-heterozygotesColumn {het_field} -siteTypeColumn {site_type_field} "
            f"{dbsnp_part} -emptyField {empty_field} "
            f"-exonicAlleleFunctionColumn {ea_fun_field} "
            f"-primaryDelimiter '{value_delim}' -fieldSeparator '{field_delim}'"
        )
