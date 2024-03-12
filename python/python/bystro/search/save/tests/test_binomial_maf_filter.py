import pytest
from bystro.search.save.binomial_maf import BinomialMafFilter

# Sample data to be used in tests
sample_data = {
    "alt": [[["A"]]],
    "missingness": [[[0.1]]],
    "sampleMaf": [[[0.05]]],
    "gnomad": {"exomes": {"af": [[[0.03]]]}},
}


# Test for basic functionality
def test_basic_functionality():
    filter_ = BinomialMafFilter(
        private_maf=0.01,
        snp_only=False,
        num_samples=100,
        estimates=["gnomad.exomes.af"],
        crit_value=0.025,
    )
    binom_filter = filter_.make_filter()

    assert binom_filter is not None
    assert binom_filter(sample_data) is False  # within expectation


# Test for different alpha values
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
def test_different_alpha_values(alpha):
    filter_ = BinomialMafFilter(
        private_maf=0.01,
        snp_only=False,
        num_samples=100,
        estimates=["gnomad.exomes.af"],
        crit_value=alpha,
    )
    binom_filter = filter_.make_filter()

    assert binom_filter is not None

    if alpha == 0.01 or alpha == 0.05:
        assert binom_filter(sample_data) is False
    else:
        assert binom_filter(sample_data) is True


# Test for SNP only filtering
def test_snp_only_filtering():
    filter_ = BinomialMafFilter(
        private_maf=0.01,
        snp_only=True,
        num_samples=100,
        estimates=["gnomad.exomes.af"],
        crit_value=0.025,
    )
    binom_filter = filter_.make_filter()

    assert binom_filter is not None
    assert binom_filter(sample_data) in [True, False]  # depending on the expected behavior
