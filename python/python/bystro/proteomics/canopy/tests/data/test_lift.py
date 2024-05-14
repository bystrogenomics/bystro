from itertools import combinations

import pytest

from bystro.proteomics.canopy.data.lift import LiftData, check_substrings


def test_lift_data_all_positives():
    matrices = tuple(LiftData._supported_matrices)
    versions = tuple(LiftData._version_map.keys())
    version_size = {'v5.0': 11083, 'v4.1': 7596, 'v4.0': 5284}
    nc2 = [x for x in combinations(versions, 2)]
    n_choose_3 = [list(x) + [y] for y in matrices for x in nc2]
    for x in n_choose_3:
        if x[0] != x[1]:
            # the case where the user attempts to lift to the existing version is a failure tested below.
            ld = LiftData(*x)
            assert ld.scale_factors.shape[0] == version_size[x[0]]
            assert ld.scale_factors.shape[0] == version_size[x[0]]


def test_lift_data_type_erros():
    # if we feed the wrong data types we get informative errrors:
    with pytest.raises(TypeError):
        LiftData(5.0, 'v4.0', 'EDTA Plasma')
    with pytest.raises(TypeError):
        LiftData('v5.0', 4.0, 'EDTA Plasma')
    with pytest.raises(TypeError):
        LiftData('v5.0', 'v4.0', True)


def test_lift_data_negative():
    # these are all failure states:
    # fails on not an assay version:
    with pytest.raises(ValueError):
        LiftData('v5.0', 'v99.99', 'EDTA Plasma')
    # fails on lifting to self:
    with pytest.raises(ValueError):
        LiftData('v5.0', 'v5.0', 'EDTA Plasma')
    with pytest.raises(ValueError):
        LiftData('v5.0', 'v4.1', 'Girble Tears')
    with pytest.raises(ValueError):
        LiftData('v5.0', 'v4.1', 'Urine')
    with pytest.raises(ValueError):
        LiftData('v5.0', 'v4.1', 'CSF')
    with pytest.raises(ValueError):
        LiftData('v5.0', 'v4.1', 'other')


def test_check_substrings():
    input_text = 'this is the test'
    sub1 = 'this'
    sub2 = 'test'
    assert check_substrings(input_text, sub1, sub2)
    assert not check_substrings(input_text, sub2, sub1)
    assert not check_substrings(input_text, 'not', sub2)
    assert not check_substrings(input_text, sub1, 'not')
