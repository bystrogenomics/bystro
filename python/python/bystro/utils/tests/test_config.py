from unittest.mock import patch

import pytest

from bystro.utils.config import (
    BYSTRO_PROJECT_ROOT,
    BYSTRO_CONFIG_DIR,
    get_opensearch_config,
    _get_bystro_project_root,
    get_mapping_config,
    ReferenceGenome,
)


def test_get_bystro_project_root():
    expected_startup_yml_path = BYSTRO_PROJECT_ROOT / "startup.yml"
    assert expected_startup_yml_path.exists()


@patch("pathlib.Path.glob", return_value=[])
def test_get_bystro_project_root_error_case(mocked_glob):  # noqa: ARG001  (arg is actually necessary)
    """Test case where Path.glob never finds startup.yml in any directory it searches."""
    with pytest.raises(FileNotFoundError, match="this is a bug"):
        _get_bystro_project_root()

@patch("bystro.utils.config.OPENSEARCH_CONFIG_PATH", BYSTRO_CONFIG_DIR / "opensearch.clean.yml")
def test_get_opensearch_config():
    opensearch_config = get_opensearch_config()
    expected_keys = {"connection", "auth"}
    assert expected_keys == set(opensearch_config.keys())


def test_get_mapping_config():
    assert 4 == len(get_mapping_config(ReferenceGenome.hg38))
    assert 4 == len(get_mapping_config(ReferenceGenome.hg19))
