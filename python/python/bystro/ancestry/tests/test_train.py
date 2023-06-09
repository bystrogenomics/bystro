"""Tests for ancestry model training code."""

from bystro.ancestry import train


## this is not a very useful test, really just testing that train
## loads correctly
def test_ploidy_is_two():
    assert "Ancestry" in train.__doc__
