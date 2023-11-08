def drop_row_if_out_of_hwe(
    chi2_crit: float,
    n: float,
    missingness: float,
    sampleMaf: float,
    heterozygosity: float,
    homozygosity: float) -> bool:

    cdef float p, n_updated
    cdef float expect_hets, expect_homozygotes_ref, expect_homozygotes_alt
    cdef float hets, homozygotes_alt, homozygous_ref, test

    if missingness >= 1.0:
        return True

    p = 1.0 - sampleMaf
    n_updated = n * (1.0 - missingness)

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