# # conftest.py

# import pytest


# def pytest_addoption(parser):
#     parser.addoption(
#         # "--run_vpn_integation_tests",
#         "--foo",
#         action="store_true",
#         default=False,
#         help="run tests requiring vpn access",
#     )


# def pytest_configure(config):
#     config.addinivalue_line("markers", "vpn_integration_test: mark test as requiring vpn")


# def pytest_collection_modifyitems(config, items):
#     if config.getoption("--run_vpn_integration_tests"):
#         return
#     vpn_integration_test = pytest.mark.skip(reason="need --run_vpn_integration_test option to run")
#     for item in items:
#         if "vpn_integration" in item.keywords:
#             item.add_marker(skip_slow)


# conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run_vpn_integration_tests", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "vpn_integration_test: mark test as requiring VPN access to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_vpn_integration_tests"):
        #  # if we get the CLI flag, don't turn off any tests...
        return
    skip_vpn_integration_test = pytest.mark.skip(reason="need --run_vpn_integration_tests option to run")
    for item in items:
        if "vpn_integration_test" in item.keywords:
            item.add_marker(skip_vpn_integration_test)
