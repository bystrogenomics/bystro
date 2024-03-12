from unittest.mock import patch

from bystro.search.save.hwe import HWEFilter


def test_out_of_hwe_proportions():
    fails_filter = HWEFilter(num_samples=100, crit_value=0.05).make_filter()

    assert fails_filter is not None

    # A dummy doc
    doc = {
        "sampleMaf": [[[0.5]]],
        "missingness": [[[0]]],
        "heterozygosity": [[[0.99]]],
        "homozygosity": [[[0.01]]],
    }

    # Expected proportions based on 0.5 maf:
    #  expected hets are:  2 * p * (1 - p) * n = 50
    #  expected homozygous reference = (p**2) * n = (0.5**2) * 100 = 25
    #  expected homozygous alternate =  100 - (50 + 25) = 25 # noqa: ERA001

    # With 100 samples and .9999 heterozygosity, we have 99 hets, 1 homozygote, and no reference
    # our chi2 test statistic is:
    # (((99 - 50) ** 2) / 50) +
    # (((1 - 25) ** 2) / 25) +
    # (((0 - 25) ** 2) / 25) # noqa: ERA001
    # = 96.06
    # which is greater than our critical value of 3.84, so we should fail the row

    res = fails_filter(doc)

    assert res is True


def test_in_hwe_proportions():
    fails_filter = HWEFilter(num_samples=100, crit_value=0.05).make_filter()

    assert fails_filter is not None

    doc = {
        "sampleMaf": [[[0.5]]],
        "missingness": [[[0]]],
        "heterozygosity": [[[0.5]]],
        "homozygosity": [[[0.25]]],
    }

    # With heterozygosity at .25, we have 25 hets, 25 homozygotes, and 50 reference,
    # which is the expected proportion, so we shouldn't fail

    res = fails_filter(doc)

    assert res is False


def test_makeHweFilter_sampleMaf_zero():
    fails_filter = HWEFilter(num_samples=10, crit_value=0.05).make_filter()

    assert fails_filter is not None

    doc = {
        "sampleMaf": [[[0]]],
        "missingness": [[[0]]],
        "heterozygosity": [[[0.1]]],
        "homozygosity": [[[0.2]]],
    }

    # When sampleMaf is 0, we skip the site
    res = fails_filter(doc)

    assert res is False


@patch("bystro.search.save.hwe.logger.warning")
def test_makeHweFilter_low_num_samples(mock_log_warning):
    # Check if it logs warning and returns None for num_samples < 1
    result = HWEFilter(num_samples=0, crit_value=0.05).make_filter()
    mock_log_warning.assert_called_once_with(
        "To perform the HWE filter, number of samples must be greater than 0, got %s", 0
    )
    assert result is None
