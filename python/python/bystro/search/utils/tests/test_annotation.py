import pytest

from bystro.search.utils.annotation import (
    DelimitersConfig,
    get_delimiters,
    get_config_file_path,
    StatisticsConfig,
    StatisticsOutputExtensions,
    Statistics,
)


def assert_defaults(config: DelimitersConfig):
    assert config.field == "\t"
    assert config.position == "|"
    assert config.overlap == chr(31)
    assert config.value == ";"
    assert config.empty_field == "!"


def test_default_values():
    config = DelimitersConfig()
    assert_defaults(config)


def test_from_dict_no_arg():
    config = DelimitersConfig.from_dict()
    assert_defaults(config)


def test_from_dict_no_delimiters_key():
    config_dict = {"random_key": "random_value"}
    config = DelimitersConfig.from_dict(config_dict)
    assert_defaults(config)


def test_from_dict_with_delimiters_key():
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


def test_from_dict_partial_delimiters_key():
    config_dict = {"delimiters": {"field": "x", "position": "y"}}
    config = DelimitersConfig.from_dict(config_dict)
    assert config.field == "x"
    assert config.position == "y"
    assert config.overlap == chr(31)  # default value
    assert config.value == ";"  # default value
    assert config.empty_field == "!"  # default value


def test_get_delimiters_default():
    result = get_delimiters()
    assert result == {
        "field": "\t",
        "position": "|",
        "overlap": chr(31),
        "value": ";",
        "empty_field": "!",
    }


def test_get_delimiters_none_input():
    result = get_delimiters(None)
    assert result == {
        "field": "\t",
        "position": "|",
        "overlap": chr(31),
        "value": ";",
        "empty_field": "!",
    }


def test_get_delimiters_with_input():
    config_dict = {
        "delimiters": {
            "field": "x",
            "position": "y",
            "overlap": "z",
            "value": "w",
            "empty_field": "v",
        }
    }
    result = get_delimiters(config_dict)
    assert result == {
        "field": "x",
        "position": "y",
        "overlap": "z",
        "value": "w",
        "empty_field": "v",
    }


def test_get_delimiters_partial_input():
    config_dict = {"delimiters": {"field": "x", "position": "y"}}
    result = get_delimiters(config_dict)
    assert result == {
        "field": "x",
        "position": "y",
        "overlap": chr(31),  # default value
        "value": ";",  # default value
        "empty_field": "!",  # default value
    }


def test_get_config_file_path_no_path_found(mocker):
    mocker.patch(
        "bystro.search.utils.annotation.glob", return_value=[]
    )  # Change `your_module` to the actual module name
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
    assert config.dbSNPnameField == "dbSNP.name"
    assert config.siteTypeField == "refSeq.siteType"
    assert config.exonicAlleleFunctionField == "refSeq.exonicAlleleFunction"
    assert config.refField == "ref"
    assert config.homozygotesField == "homozygotes"
    assert config.heterozygotesField == "heterozygotes"
    assert config.altField == "alt"
    assert config.programPath == "bystro-stats"
    assert isinstance(config.outputExtensions, StatisticsOutputExtensions)


def test_statistics_config_from_dict_none():
    config = StatisticsConfig.from_dict()
    assert (
        config.dbSNPnameField == "dbSNP.name"
    )  # Check one of the default values as a representative


def test_statistics_config_from_dict_no_statistics_key():
    annotation_config = {"random_key": "random_value"}
    config = StatisticsConfig.from_dict(annotation_config)
    assert (
        config.dbSNPnameField == "dbSNP.name"
    )  # Again, just checking one representative default value


def test_statistics_config_from_dict_with_statistics_key():
    annotation_config = {
        "statistics": {
            "dbSNPnameField": "new.dbSNP.name",
            "siteTypeField": "new.refSeq.siteType",
            "outputExtensions": {
                "json": ".new.json",
                "tab": ".new.tsv",
                "qc": ".new.qc.tsv",
            },
        }
    }
    config = StatisticsConfig.from_dict(annotation_config)
    assert config.dbSNPnameField == "new.dbSNP.name"
    assert config.siteTypeField == "new.refSeq.siteType"
    assert config.outputExtensions.json == ".new.json"
    assert config.outputExtensions.tsv == ".new.tsv"
    assert config.outputExtensions.qc == ".new.qc.tsv"


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
        "-dbSnpNameColumn dbSNP.name -emptyField '!' "
        "-exonicAlleleFunctionColumn refSeq.exonicAlleleFunction "
        "-primaryDelimiter ';' -fieldSeparator '\t'"
    )

    assert statistics.stdin_cli_stats_command == expected_args
