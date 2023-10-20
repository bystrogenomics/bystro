"""Configure pytest."""

import pytest

# we define a pytest marker `integration` that disables a test by
# default unless pytest is run with the long flag
# --run_integration_tests


def pytest_addoption(parser):
    parser.addoption(
        "--run_integration_tests", action="store_true", default=False, help="run integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_integration_tests"):
        #  # if we get the CLI flag, don't turn off any tests...
        return
    skip_integration_test = pytest.mark.skip(reason="need --run_integration_tests option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration_test)
