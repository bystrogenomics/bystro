"""Tests for ancestry model training code."""

from bystro.ancestry import train


## this is not a very useful test, really just testing that train
## loaded successfully, didn't fail on import


def test_module_is_loaded_correctly():
    assert "Ancestry" in train.__doc__
