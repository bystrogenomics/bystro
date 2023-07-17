"""Test train utilities."""
import pytest

from bystro.ancestry.train_utils import is_autosomal_variant


def test_is_autosomal_variant():
    assert is_autosomal_variant("chr1:123456:A:T")
    assert is_autosomal_variant("chr22:1:T:A")
    assert not is_autosomal_variant("22:1:A:G")
    assert not is_autosomal_variant("chrX:1:A:G")
    assert not is_autosomal_variant("chr23:1:G:C")
    assert not is_autosomal_variant("chr22:1:A:")
    assert not is_autosomal_variant("chr22:1:A:AT")
    assert not is_autosomal_variant("chr22:1:GC:AT")
    assert not is_autosomal_variant("chr22:1:X:Y")
    with pytest.raises(ValueError, match="cannot have identical ref and alt alleles"):
        is_autosomal_variant("chr22:1:A:A")
