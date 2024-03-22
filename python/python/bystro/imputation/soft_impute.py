import numpy as np
import numpy.linalg as la
from sklearn.utils.extmath import randomized_svd
from tqdm import trange

from bystro.impute._base import BaseImpute

F32PREC = np.finfo(np.float32).eps


class SoftImpute(BaseImpute):
    def __init__(
        self,
        shrinkage_value=.1,
        training_options=None,
        init_fill_method="zero",
    ):
        if training_options is None:
            training_options = {}
        super().__init__(
            self,
            fill_method=init_fill_method,
            training_options=training_options,
        )
        self.shrinkage_value = shrinkage_value

    def _converged(self, X_old, X_new, missing_mask):
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference**2)
        old_norm = np.sqrt((old_missing_values**2).sum())
        if old_norm == 0 or (old_norm < F32PREC and np.sqrt(ssd) > F32PREC):
            return False
        return (np.sqrt(ssd) / old_norm) < self.training_options[
            "convergence_threshold"
        ]

    def _svd_step(self, X, shrinkage_value, max_rank=None):
        training_options = self.training_options
        if max_rank:
            (U, s, V) = randomized_svd(
                X,
                max_rank,
                n_iter=training_options["n_power_iterations"],
                random_state=None,
            )
        else:
            (U, s, V) = la.svd(X, full_matrices=False, compute_uv=True)
        s_thresh = np.maximum(s - shrinkage_value, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        S_thresh = np.diag(s_thresh)
        X_reconstruction = np.dot(U_thresh, np.dot(S_thresh, V_thresh))
        return X_reconstruction, rank

    def _solve(self, X, missing_mask, progress_bar=True):
        training_options = self.training_options

        X_filled = X

        for i in trange(
            training_options["n_iterations"], disable=not progress_bar
        ):
            X_reconstruction, rank = self._svd_step(
                X_filled, self.shrinkage_value, max_rank=training_options["max_rank"]
            )

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstruction,
                missing_mask=missing_mask,
            )
            X_filled[missing_mask] = X_reconstruction[missing_mask]
            if converged:
                break

        return X_filled

    def _fill_training_options(self, training_options):
        default_options = {
            "n_iterations": 100,
            "convergence_threshold": 0.001,
            "n_power_iterations": 1,
            "max_rank": None,
        }
        tops = {**default_options, **training_options}

        default_keys = set(default_options.keys())
        final_keys = set(tops.keys())

        expected_but_missing_keys = default_keys - final_keys
        unexpected_but_present_keys = final_keys - default_keys
        if expected_but_missing_keys:
            raise ValueError("training options were expected but not found")
        if unexpected_but_present_keys:
            raise ValueError("training options were unrecognized but provided")

        return tops
