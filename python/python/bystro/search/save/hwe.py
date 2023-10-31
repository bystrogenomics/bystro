import logging
from typing import Callable

from msgspec import Struct
from scipy.stats import chi2
from bystro.search.save.c_hwe import drop_row_if_out_of_hwe

logger = logging.getLogger(__name__)

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
    ) -> Callable[[object], bool] | None:
        if self.num_samples <= 0:
            logger.warning(
                "To perform the HWE filter, number of samples must be greater than 0, got %s",
                self.num_samples
            )
            return None

        chi2_crit = chi2.isf(self.crit_value, 1)

        n_samples = float(self.num_samples)

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
