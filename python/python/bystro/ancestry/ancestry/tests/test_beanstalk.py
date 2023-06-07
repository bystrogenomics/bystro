"""Test ancestry/beanstalk.py."""

import pytest
from pydantic import ValidationError

from ancestry.beanstalk import Address


def test_Address() -> None:
    """Ensure we can instantiate, validate Address correctly."""
    well_formed_addresses = [
        "127.0.0.1:80",
        "//127.0.0.1:80",
        "beanstalkd://127.0.0.1:80",
    ]
    for raw_address in well_formed_addresses:
        address = Address.from_str(raw_address)
        assert address.host == "127.0.0.1"
        assert address.port == 80

    malformed_addresses = [
        "127.0.0.0.1:80",
        "127.0.1:80",
        "127.0.0.1",
        "///127.0.0.1:80",
        "beanstalkd//127.0.0.1:80",
    ]
    for raw_address in malformed_addresses:
        with pytest.raises((ValidationError, ValueError)):
            address = Address.from_str(raw_address)
