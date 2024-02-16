import numpy as np

from bystro.search.save.hwe import HWEFilter

N = 100000

# Number of random samples to test for each parameter
SAMPLE_SIZE = 50000

# Generate random sample values
rng = np.random.default_rng()  # Create a random number generator instance
missingness_values = list(rng.random(SAMPLE_SIZE))
sample_maf_values = list(rng.random(SAMPLE_SIZE))
heterozygosity_values = list(rng.random(SAMPLE_SIZE))
homozygosity_values = list(rng.random(SAMPLE_SIZE))

missingness_idx = 0
sample_maf_idx = 1
heterozygosity_idx = 2
homozygosity_idx = 3
header_row = b"missingness\tsampleMaf\theterozygosity\thomozygosity".split(b"\t")
docs = []
for missingness, sample_maf, heterozygosity, homozygosity in zip(
    missingness_values, sample_maf_values, heterozygosity_values, homozygosity_values
):
    docs.append(
        [
            bytes(f"{missingness}", "utf-8"),
            bytes(f"{sample_maf}", "utf-8"),
            bytes(f"{heterozygosity}", "utf-8"),
            bytes(f"{homozygosity}", "utf-8"),
        ]
    )


def naive_drop_row_if_out_of_hwe(
    chi2_crit: float,
    n: float,
    missingness: float,
    sample_maf: float,
    heterozygosity: float,
    homozygosity: float,
) -> bool:
    if missingness >= 1.0:
        return True

    p = 1 - sample_maf

    n_updated = n * (1 - missingness)

    expect_hets = 2 * p * (1 - p) * n_updated
    expect_homozygotes_ref = (p**2) * n_updated
    expect_homozygotes_alt = n_updated - (expect_hets + expect_homozygotes_ref)

    if expect_hets <= 0.0 or expect_homozygotes_ref <= 0.0 or expect_homozygotes_alt <= 0.0:
        return False

    hets = n_updated * heterozygosity
    homozygotes_alt = n_updated * homozygosity
    homozygous_ref = n_updated - (hets + homozygotes_alt)

    test = (
        (((hets - expect_hets) ** 2) / expect_hets)
        + (((homozygous_ref - expect_homozygotes_ref) ** 2) / expect_homozygotes_ref)
        + (((homozygotes_alt - expect_homozygotes_alt) ** 2) / expect_homozygotes_alt)
    )

    return test > chi2_crit


def test_drop_row_hwe(benchmark):
    filter_fn = HWEFilter(num_samples=N, crit_value=0.025).make_filter(header_row)

    def loop():
        assert filter_fn is not None

        for doc in docs:
            res = filter_fn(doc)

            assert res is True or res is False

    benchmark(loop)


def test_naive_drop_row_hwe(benchmark):
    num_samples = N
    crit_value = 0.025

    # Use the benchmark fixture
    def loop():
        for doc in docs:
            res = naive_drop_row_if_out_of_hwe(
                crit_value,
                num_samples,
                float(doc[missingness_idx]),
                float(doc[sample_maf_idx]),
                float(doc[heterozygosity_idx]),
                float(doc[homozygosity_idx])
            )
            assert res is True or res is False

    benchmark(loop)
