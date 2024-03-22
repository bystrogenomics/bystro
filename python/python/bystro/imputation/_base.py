import numpy as np
from sklearn.utils import check_array


class BaseImpute(object):
    def __init__(self, fill_method="zero", training_options=None):
        self.fill_method = fill_method
        self._training_options = self._fill_training_options(training_options)

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            if np.all(np.isnan(fill_values)):
                fill_values = 0
            X[missing_col, col_idx] = fill_values

    def _fill(self, X, missing_mask, fill_method=None):
        X = check_array(X, force_all_finite=False)

        X = X.copy()

        if not fill_method:
            fill_method = self.fill_method

        if fill_method == "zero":
            X[missing_mask] = 0
        elif fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        else:
            raise ValueError("Unrecognized fill method %s"%fill_method)
        return X

    def fit_transform(self, X):
        self._test_inputs()
        X, missing_mask = self._transform_training_data(X)
        observed_mask = ~missing_mask

        X_filled = self._fill(X, missing_mask)
        X_result = self._solve(X_filled, missing_mask)
        X_result[observed_mask] = X[observed_mask]
        return X_result

    def _test_inputs(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError(
                "Expected NumPy array input but got %s"
                % (type(X))
            )
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

        X_mask = np.isnan(X)
        miss_column = np.mean(X_mask,axis=0)
        miss_row = np.mean(X_mask,axis=1)

        if np.any(miss_column==1):
            raise ValueError("Entire column unobserved")
        if np.any(miss_row==1):
            raise ValueError("Entire row unobserved")

    def _transform_training_data(self, X):
        X = check_array(X, force_all_finite=False)

        self._check_input(X)
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)
        return X, missing_mask


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This implementation was based on fancyimpute, by Alex Rubinsteyn
#
