import numpy as np
from bystro.covariance.test import (
    is_square,
    is_sym,
    is_skew_sym,
    is_hankel,
    is_real,
    is_real_type,
    is_hermitian,
    is_pos_def,
    is_pos_semi_def,
    is_sym_pos_def,
    is_sym_pos_semi_def,
    is_herm_pos_def,
    is_herm_pos_semi_def,
)


def test_is_square():
    assert is_square(np.array([[1, 2], [3, 4]])) is True
    assert is_square(np.array([[1, 2, 3], [4, 5, 6]])) is False


def test_is_sym():
    assert is_sym(np.array([[1, 2], [2, 1]])) is True
    assert is_sym(np.array([[1, 2], [3, 4]])) is False


def test_is_skew_sym():
    assert is_skew_sym(np.array([[0, -2], [2, 0]])) is True
    assert is_skew_sym(np.array([[0, 1], [1, 0]])) is False


def test_is_hankel():
    assert is_hankel(np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])) is True
    assert is_hankel(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])) is False


def test_is_real():
    assert is_real(np.array([[1, 2], [3, 4]], dtype=np.complex128)) is True
    assert (
        is_real(np.array([[1, 2], [3, 4]], dtype=np.complex128) + 1j) is False
    )


def test_is_real_type():
    assert is_real_type(np.array([[1, 2], [3, 4]])) is True
    assert (
        is_real_type(np.array([[1, 2], [3, 4]], dtype=np.complex128)) is False
    )


def test_is_hermitian():
    assert is_hermitian(np.array([[1, 1j], [-1j, 1]])) is True
    assert is_hermitian(np.array([[1, 1j], [1j, 1]])) is False


def test_is_pos_def():
    assert is_pos_def(np.array([[2, 1], [1, 2]])) is True
    assert is_pos_def(np.array([[0, 1], [1, 0]])) is False


def test_is_pos_semi_def():
    assert is_pos_semi_def(np.array([[1, 0], [0, 0]])) is True
    assert is_pos_semi_def(np.array([[1, 0], [0, -1]])) is False


def test_is_sym_pos_def():
    assert is_sym_pos_def(np.array([[2, 1], [1, 2]])) is True
    assert is_sym_pos_def(np.array([[1, 2], [2, 1]])) is False


def test_is_sym_pos_semi_def():
    assert is_sym_pos_semi_def(np.array([[1, 0], [0, 0]])) is True
    assert is_sym_pos_semi_def(np.array([[1, 2], [2, 1]])) is False


def test_is_herm_pos_def():
    assert is_herm_pos_def(np.array([[2, 1j], [-1j, 2]])) is True
    assert is_herm_pos_def(np.array([[1, 1j], [1j, 1]])) is False


def test_is_herm_pos_semi_def():
    assert (
        is_herm_pos_semi_def(np.array([[1, 0], [0, 0]], dtype=np.complex128))
        is True
    )
    assert is_herm_pos_semi_def(np.array([[1, 1j], [1j, 1]])) is False
