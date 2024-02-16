from typing import Callable
import math

from msgspec import Struct
from scipy.stats import norm  # type: ignore


class BinomialMafFilter(
    Struct,
    frozen=True,
    tag="binomMaf",
    tag_field="key",
    forbid_unknown_fields=True,
    rename="camel",
):
    """
    Filter out rows that are extreme outliers, comparing the in-sample estimate of allele frequency,
    the `sampleMaf` from the indexed annotation, to the "true" population allele frequency estimate
    from the `estimates` parameter.

    This filters operates on the assumption that the alleles are binomially distributed,
    and uses the normal approximation to the binomial distribution to identify outliers.

    A variant is considered an outlier if the `sampleMaf` is outside the `1-(critValue*2)`
    rejection region. A 2 tailed test is used.

    Since very rare mutations we may have unreliable population estimates, if the in-sample
    allele frequency `sampleMaf` is less than the `privateMaf` threshold,
    we retain the variant regardless of whether it is an outlier.

    Parameters
    ----------
    private_maf : float
        The rare frequency threshold. If the sampleMaf is less than this value, the variant is retained
    snp_only : bool
        Whether or not only single nucleotide variants (SNPs) should be considered
    num_samples : int
        The number of samples in the population
    estimates : list[str]
        The names of the columns that contain the population allele frequency estimates
    crit_value : float, optional
        The critical value defining the rejection region. A 2 tail test is used,
        so the rejection region is `1-(critValue*2)`.
        Default: 0.025
    """

    private_maf: float
    snp_only: bool
    num_samples: int
    estimates: list[str]
    crit_value: float | None = 0.025

    def make_filter(self, header_fields: list[bytes]) -> Callable[[list[bytes]], bool] | None:
        private_maf = self.private_maf
        snp_only = self.snp_only
        num_samples = self.num_samples
        estimates = self.estimates
        alpha = self.crit_value

        total_alleles = num_samples * 2

        if total_alleles < 2:
            return None

        if not estimates:
            return None

        if not alpha or alpha == 0 or alpha > 0.5:
            print("Error: Alpha must be larger than 0 and smaller than 0.5")
            return None

        z_crit = norm.ppf(1 - alpha)

        private_maf = private_maf or 0
        min_possible_estimate = round(1 / total_alleles, 2)

        if private_maf < min_possible_estimate:
            private_maf = min_possible_estimate

        estimate_field_indices = []
        for e in estimates:
            idx = header_fields.index(bytes(e, "utf-8"))
            if idx == -1:
                raise ValueError(f"Estimate field {e} not found in header fields")
            estimate_field_indices.append(idx)

        missingness_idx = header_fields.index(b"missingness")
        sample_maf_idx = header_fields.index(b"sampleMaf")
        alt_idx = header_fields.index(b"alt")

        if missingness_idx == -1:
            raise ValueError("Missingness field not found in header fields")

        if sample_maf_idx == -1:
            raise ValueError("SampleMaf field not found in header fields")

        if alt_idx == -1:
            raise ValueError("Alt field not found in header fields")

        def binom_filter(row: list[bytes]):
            if snp_only and len(row[alt_idx]) > 1:
                return False

            n = total_alleles * (1.0 - float(row[missingness_idx]))

            if n == 0:
                return True

            sample_maf = float(row[sample_maf_idx])
            k = n * sample_maf
            is_rare = sample_maf <= private_maf or (n == total_alleles and k < 1.5)

            tested = 0

            for field_idx in estimate_field_indices:
                if row[field_idx] == b"NA":
                    continue

                p = float(row[field_idx])

                if (p == 1 and not is_rare) or (p <= private_maf and is_rare):
                    return False

                if abs(k - n * p) / math.sqrt(n * p * (1 - p)) <= z_crit:
                    return False

                tested += 1

            # Very rare mutations may be private to the sample, so we don't want to filter them out
            if tested == 0 and is_rare:
                return False

            return True

        return binom_filter
