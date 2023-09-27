from bystro.utils.config import get_bystro_project_root, get_opensearch_config


def test_get_bystro_project_root():
    bystro_project_root = get_bystro_project_root()
    assert "bystro" in str(bystro_project_root)


def test_get_opensearch_config():
    opensearch_config = get_opensearch_config()
    expected_keys = set(["connection", "auth"])
    assert expected_keys == set(opensearch_config.keys())
