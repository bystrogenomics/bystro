import logging
from typing import Callable

from msgspec import Struct
import jax
import jax.numpy as jnp
from scipy.stats import chi2

logger = logging.getLogger(__name__)


@jax.jit
def drop_row_if_out_of_hwe(
    chi2_crit: float,
    n: float,
    missingness: float,
    sampleMaf: float,
    heterozygosity: float,
    homozygosity: float,
) -> jax.Array:
    def true(_):
        return True

    def false(_):
        return False

    def continue_calculation(_):
        p = 1 - sampleMaf

        n_updated = n * (1 - missingness)

        def p_nonzero(_):
            expect_hets = 2 * p * (1 - p) * n_updated
            expect_homozygotes_ref = (p**2) * n_updated
            expect_homozygotes_alt = n_updated - (expect_hets + expect_homozygotes_ref)

            def expected_zero(_):
                return False

            def expected_nonzero(_):
                hets = n_updated * heterozygosity
                homozygotes_alt = n_updated * homozygosity
                homozygous_ref = n_updated - (hets + homozygotes_alt)

                test = (
                    (((hets - expect_hets) ** 2) / expect_hets)
                    + (
                        ((homozygous_ref - expect_homozygotes_ref) ** 2)
                        / expect_homozygotes_ref
                    )
                    + (
                        ((homozygotes_alt - expect_homozygotes_alt) ** 2)
                        / expect_homozygotes_alt
                    )
                )
                return test > chi2_crit

            return jax.lax.cond(
                jnp.any(
                    jnp.array(
                        [expect_hets, expect_homozygotes_ref, expect_homozygotes_alt]
                    )
                    == 0
                ),
                expected_zero,
                expected_nonzero,
                None,
            )

        return jax.lax.cond(p == 0, false, p_nonzero, None)

    return jax.lax.cond(missingness == 1, true, continue_calculation, None)


class HWEFilter(
    Struct,
    frozen=True,
    tag="hwe",
    tag_field="key",
    forbid_unknown_fields=True,
    rename="camel",
):
    """
    A Hardy-Weinberg Equilibrium (HWE) filter,
    which filters out variants that are not in HWE.

    Parameters
    ----------
    num_samples : int
        Number of samples in the population
    crit_value : float, optional
        The critical value for the chi-squared test.
        Default: 0.025
    """

    num_samples: int
    crit_value: float | None = 0.025

    def make_filter(
        self,
    ) -> Callable[[object], jax.Array] | None:
        if self.num_samples <= 0:
            logger.warning(
                "To perform the HWE filter, number of samples must be greater than 0, got %s",
                self.num_samples
            )
            return None

        chi2_crit = chi2.isf(self.crit_value, 1)

        n_samples = float(self.num_samples)

        @jax.jit
        def filter_function(doc: object):
            return drop_row_if_out_of_hwe(
                chi2_crit=chi2_crit,
                n=n_samples,
                missingness=doc["missingness"][0][0],
                sampleMaf=doc["sampleMaf"][0][0],
                heterozygosity=doc["heterozygosity"][0][0],
                homozygosity=doc["homozygosity"][0][0],
            )

        return filter_function
