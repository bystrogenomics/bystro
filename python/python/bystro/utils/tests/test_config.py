from unittest.mock import patch

import pytest
from bystro.utils.config import get_bystro_project_root, get_opensearch_config


def test_get_bystro_project_root():
    bystro_project_root = get_bystro_project_root()
    assert "bystro" in str(bystro_project_root)


@patch("pathlib.Path.glob", return_value=[])
def test_get_bystro_project_root_error_case(mocked_glob):  # noqa: ARG001  (arg is actually necessary)
    """Test case where Path.glob never returns startup.yml."""
    with pytest.raises(FileNotFoundError, match="this is a bug"):
        get_bystro_project_root()


def test_get_opensearch_config():
    opensearch_config = get_opensearch_config()
    expected_keys = {"connection", "auth"}
    assert expected_keys == set(opensearch_config.keys())
