from msgspec import json
import pytest

from bystro.search.utils.annotation import (
    DelimitersConfig,
    get_config_file_path,
    StatisticsConfig,
    StatisticsOutputExtensions,
    Statistics,
)


def assert_defaults(config: DelimitersConfig):
    assert config.field == "\t"
    assert config.position == "|"
    assert config.overlap == "/"
    assert config.value == ";"
    assert config.empty_field == "NA"


def test_delimiters_default_values():
    config = DelimitersConfig()
    assert_defaults(config)


def test_delimiters_from_dict_no_arg():
    with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'annotation_config'"):
        # Ignoring type checking because we're testing the error
        DelimitersConfig.from_dict()  # type: ignore


def test_delimiters_from_dict_no_delimiters_key():
    config_dict = {"random_key": "random_value"}
    config = DelimitersConfig.from_dict(config_dict)
    assert_defaults(config)


def test_delimiters_from_dict_unexpected_key():
    annotation_config = {"delimiters": {"random_delim_key": "random_value"}}
    with pytest.raises(TypeError, match=r"Unexpected keyword argument 'random_delim_key'"):
        DelimitersConfig.from_dict(annotation_config)


def test_delimiters_from_dict_with_delimiters_key():
    config_dict = {
        "delimiters": {
            "field": "x",
            "position": "y",
            "overlap": "z",
            "value": "w",
            "empty_field": "v",
        }
    }
    config = DelimitersConfig.from_dict(config_dict)
    assert config.field == "x"
    assert config.position == "y"
    assert config.overlap == "z"
    assert config.value == "w"
    assert config.empty_field == "v"


def test_delimiters_from_dict_partial_delimiters_key():
    config_dict = {"delimiters": {"field": "x", "position": "y"}}
    config = DelimitersConfig.from_dict(config_dict)
    assert config.field == "x"
    assert config.position == "y"
    assert config.overlap == "/"  # default value
    assert config.value == ";"  # default value
    assert config.empty_field == "NA"  # default value


def test_delimiters_unexpected_key():
    with pytest.raises(TypeError, match=r"Unexpected keyword argument 'random_delim_key2'"):
        # Ignoring type checking because we're testing the error
        DelimitersConfig(random_delim_key2="random_value")  # type: ignore


def test_get_config_file_path_no_path_found(mocker):
    mocker.patch("bystro.search.utils.annotation.glob", return_value=[])

    with pytest.raises(ValueError, match=r"No config path found for the assembly"):
        get_config_file_path("/dummy/path", "dummy_assembly")


def test_get_config_file_path_single_path_found(mocker):
    mocked_path = "/dummy/path/dummy_assembly.yml"
    mocker.patch("bystro.search.utils.annotation.glob", return_value=[mocked_path])
    result = get_config_file_path("/dummy/path", "dummy_assembly")
    assert result == mocked_path


def test_get_config_file_path_single_path_found_with_extension(mocker):
    mocked_path = "/dummy/path/dummy_assembly.blargh"
    mocker.patch("bystro.search.utils.annotation.glob", return_value=[mocked_path])
    result = get_config_file_path("/dummy/path", "dummy_assembly", suffix=".blargh")
    assert result == mocked_path


def test_get_config_file_path_single_path_found_with_extension_wildcard(mocker):
    mocked_path = "/dummy/path/dummy_assembly.blargh"
    mocker.patch("bystro.search.utils.annotation.glob", return_value=[mocked_path])
    result = get_config_file_path("/dummy/path", "dummy_assembly", suffix=".bl*rgh")
    assert result == mocked_path


def test_get_config_file_path_multiple_paths_found(mocker, capsys):
    mocked_paths = [
        "/dummy/path/dummy_assembly_1.yml",
        "/dummy/path/dummy_assembly_2.yml",
    ]
    mocker.patch("bystro.search.utils.annotation.glob", return_value=mocked_paths)
    result = get_config_file_path("/dummy/path", "dummy_assembly")
    captured = capsys.readouterr()  # Capture the standard output
    assert "More than 1 config path found, choosing first" in captured.out
    assert result == mocked_paths[0]


def test_statistics_output_extensions_defaults():
    extensions = StatisticsOutputExtensions()
    assert extensions.json == "statistics.json"
    assert extensions.tsv == "statistics.tsv"
    assert extensions.qc == "statistics.qc.tsv"


def test_statistics_config_defaults():
    config = StatisticsConfig()
    assert config.dbsnp_name_field == "dbSNP.name"
    assert config.site_type_field == "refSeq.siteType"
    assert config.exonic_allele_function_field == "refSeq.exonicAlleleFunction"
    assert config.ref_field == "ref"
    assert config.homozygotes_field == "homozygotes"
    assert config.heterozygotes_field == "heterozygotes"
    assert config.alt_field == "alt"
    assert config.program_path == "bystro-stats"
    assert isinstance(config.output_extension, StatisticsOutputExtensions)


def test_statistics_config_from_dict_none():
    with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'annotation_config'"):
        # Ignoring type checking because we're testing the error
        StatisticsConfig.from_dict()  # type: ignore


def test_statistics_config_from_dict_no_statistics_key():
    annotation_config = {"random_key": "random_value"}
    config = StatisticsConfig.from_dict(annotation_config)

    assert (
        config.dbsnp_name_field == "dbSNP.name"
    )  # Again, just checking one representative default value


def test_statistics_config_no_statistics_key():
    with pytest.raises(TypeError, match=r"Unexpected keyword argument 'random_stats_key2'"):
        StatisticsConfig(random_stats_key2="random_value")  # type: ignore


def test_statistics_config_from_dict_unexpected_key():
    annotation_config = {"statistics": {"random_key": "random_value"}}
    with pytest.raises(TypeError, match=r"Unexpected keyword argument 'random_key'"):
        StatisticsConfig.from_dict(annotation_config)


def test_statistics_config_from_dict_with_statistics_key():
    annotation_config = {
        "statistics": {
            "dbsnp_name_field": "new.dbSNP.name",
            "site_type_field": "new.refSeq.siteType",
            "output_extension": {
                "json": ".new.json",
                "tsv": ".new.tsv",
                "qc": ".new.qc.tsv",
            },
        }
    }
    config = StatisticsConfig.from_dict(annotation_config)
    assert config.dbsnp_name_field == "new.dbSNP.name"
    assert config.site_type_field == "new.refSeq.siteType"
    assert config.output_extension.json == ".new.json"
    assert config.output_extension.tsv == ".new.tsv"
    assert config.output_extension.qc == ".new.qc.tsv"


def test_statistics_init_program_not_found(mocker):
    mocker.patch("shutil.which", return_value=None)
    with pytest.raises(ValueError, match=r"Couldn't find statistics program"):
        Statistics("/dummy/path", None)


def test_statistics_get_stats_arguments(mocker):
    # Mocking which to return a valid program path
    mocker.patch("shutil.which", return_value="/path/to/bystro-stats")

    statistics = Statistics("/dummy/output_base_path", None)

    # Note: Your actual values might differ based on the defaults
    # in StatisticsConfig and DelimitersConfig
    expected_args = (
        "/path/to/bystro-stats -outJsonPath /dummy/output_base_path.statistics.json "
        "-outTabPath /dummy/output_base_path.statistics.tsv "
        "-outQcTabPath /dummy/output_base_path.statistics.qc.tsv "
        "-refColumn ref -altColumn alt -homozygotesColumn homozygotes "
        "-heterozygotesColumn heterozygotes -siteTypeColumn refSeq.siteType "
        "-dbSnpNameColumn dbSNP.name -emptyField NA "
        "-exonicAlleleFunctionColumn refSeq.exonicAlleleFunction "
        "-primaryDelimiter ';' -fieldSeparator '\t'"
    )

    assert statistics.stdin_cli_stats_command == expected_args


def test_StatisticsConfig_camel_decamel():
    annotation_config = {
        "statistics": {
            "dbsnp_name_field": "new.dbSNP.name",
            "site_type_field": "new.refSeq.siteType",
            "output_extension": {
                "json": ".new.json",
                "tsv": ".new.tsv",
                "qc": ".new.qc.tsv",
            },
        }
    }
    config = StatisticsConfig.from_dict(annotation_config)
    serialized_values = json.encode(config)
    expected_value = {
        "dbsnpNameField": "new.dbSNP.name",
        "siteTypeField": "new.refSeq.siteType",
        "exonicAlleleFunctionField": "refSeq.exonicAlleleFunction",
        "refField": "ref",
        "homozygotesField": "homozygotes",
        "heterozygotesField": "heterozygotes",
        "altField": "alt",
        "programPath": "bystro-stats",
        "outputExtension": {
            "json": ".new.json", "tsv": ".new.tsv", "qc": ".new.qc.tsv"
        }
    }
    serialized_expected_value = json.encode(expected_value)
    assert serialized_values == serialized_expected_value

    deserialized_values = json.decode(serialized_expected_value, type=StatisticsConfig)
    assert deserialized_values == config
