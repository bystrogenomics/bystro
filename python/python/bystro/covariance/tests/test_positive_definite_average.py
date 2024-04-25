"""
The methods for unit testing here come from pyriemann

Copyright (c) 2015-2024, authors of pyRiemann
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
import pytest
import numpy as np
from bystro.covariance.positive_definite_average import (
    is_positive_definite,
    pd_mean_harmonic,
    pd_mean_karcher,
)


def test_is_positive_definite():
    A = np.array([[2, 0], [0, 3]])
    B = np.array([[1, 2], [2, 1]])  # Not positive definite
    assert is_positive_definite(A) is True
    assert is_positive_definite(B) is False


def test_pd_mean_harmonic():
    A = np.array([[2, 0], [0, 3]])
    B = np.array([[4, 0], [0, 5]])
    result = pd_mean_harmonic([A, B])
    expected = np.array([[2.66667, 0], [0, 3.75]])
    np.testing.assert_array_almost_equal(result, expected, decimal=5)

    with pytest.raises(ValueError):
        pd_mean_harmonic([])  # Empty list
    with pytest.raises(ValueError):
        pd_mean_harmonic(
            [A, np.array([[1, 2], [2, 1]])]
        )  # Not all positive definite


def test_pd_mean_karcher():
    A = np.array([[3, 0], [0, 3]])
    B = np.array([[2, 0], [0, 4]])
    result = pd_mean_karcher([A, B])
    expected = np.array([[2.5, 0], [0, 3.5]])
    np.testing.assert_array_almost_equal(result, expected, decimal=5)

    with pytest.raises(ValueError):
        pd_mean_karcher([])  # Empty list

