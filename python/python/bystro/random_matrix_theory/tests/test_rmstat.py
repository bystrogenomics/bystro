import numpy as np
from bystro.random_matrix_theory.rmt_stat import (
    dtw,
    ptw,
    qtw,
    dmp,
    pmp,
    qmp,
    wishart_max_par,
    d_wishart_max,
    p_wishart_max,
    q_wishart_max,
    wishart_spike_par,
    d_wishart_spike,
    p_wishart_spike,
    q_wishart_spike,
)


def test_wishart_max_par():
    a = wishart_max_par(3, 3)
    assert np.abs(a[0] - 3.33333) < 0.01
    assert np.abs(a[1] - 1.139) < 0.01


def test_wishart_spike_par():
    print(wishart_spike_par(8, ndf=1, pdim=3))
    print(wishart_spike_par(8, ndf=2, pdim=3))
    print(wishart_spike_par(8, ndf=2, pdim=3, var=2.0))
    print(wishart_spike_par(8, ndf=2, pdim=3, var=2.0, beta=2))


def test_qmp():
    assert np.abs(qmp(0.95, 3, 2) - 2.638441) < 1e-4
    assert np.abs(qmp(0.95, 4, 3) - 2.756648) < 1e-4
    assert np.abs(qmp(0.5, 4, 3) - 0.7429527) < 1e-4
    assert np.abs(qmp(0.25, 4, 3) - 0.2647415) < 1e-4
    assert np.abs(qmp(0.01, 4, 3) - 0.02713885) < 1e-4


def test_dmp():
    assert np.abs(dmp(0.3, 2, 3) - 0.3809225) < 1e-6
    assert np.abs(dmp(0.4, 2, 3) - 0.3344779) < 1e-6
    assert np.abs(dmp(0.5, 2, 3) - 0.3001054) < 1e-6
    assert np.abs(dmp(0.3, 3, 3) - 0.55893379) < 1e-6


def test_pmp():
    assert np.abs(pmp(0.8, 4, 3) - 0.5224319) < 1e-4
    assert np.abs(pmp(0.8, 2, 3) - 0.5917404) < 1e-4
    assert np.abs(pmp(0.8, 3, 3) - 0.5498151) < 1e-4
    assert np.abs(pmp(0.01, 2, 3) - 0.3333333) < 1e-4
    assert np.abs(pmp(0.99, 2, 3) - 0.633479) < 1e-4


def test_rmp():
    pass


def test_qtw():
    assert np.abs(qtw(0.8, beta=1) + 0.165321) < 1e-4
    assert np.abs(qtw(0.8, beta=2) + 1.024968) < 1e-4
    assert np.abs(qtw(0.4, beta=1) + 1.582767) < 1e-4
    assert np.abs(qtw(0.4, beta=2) + 2.03004) < 1e-4


def test_dtw():
    assert np.abs(dtw(0.8, beta=1) - 0.08534166) < 1e-4
    assert np.abs(dtw(0.8, beta=2) - 0.01167142) < 1e-4
    assert np.abs(dtw(0.4, beta=1) - 0.1287189) < 1e-4
    assert np.abs(dtw(0.4, beta=2) - 0.029657) < 1e-4


def test_ptw():
    assert np.abs(ptw(0.8, beta=1) - 0.93614) < 1e-4
    assert np.abs(ptw(0.8, beta=2) - 0.995671) < 1e-4
    assert np.abs(ptw(0.4, beta=1) - 0.893699) < 1e-4
    assert np.abs(ptw(0.4, beta=2) - 0.9878839) < 1e-4


def test_q_wishart_max():
    assert np.abs(q_wishart_max(0.8, 3, 2) - 2.449359) < 1e-4


def test_d_wishart_max():
    assert np.abs(d_wishart_max(0.8, 3, 2) - 0.2907871) < 1e-4


def test_p_wishart_max():
    assert np.abs(p_wishart_max(0.8, 3, 2) - 0.3560258) < 1e-4


def test_q_wishart_spike():
    print(q_wishart_spike(0.4, 8, ndf=3, pdim=2))


def test_d_wishart_spike():
    assert np.abs(d_wishart_spike(0.3, 5.0, ndf=1, pdim=3) - 0.02532702) < 1e-4
    assert np.abs(d_wishart_spike(0.8, 10.0, ndf=1, pdim=3) - 0.01766141) < 1e-4


def test_p_wishart_spike():
    assert np.abs(p_wishart_spike(-1.9, 10.0, ndf=1, pdim=3) - 0.1451744) < 1e-4
    assert np.abs(p_wishart_spike(0.8, 10.0, ndf=1, pdim=3) - 0.1891236) < 1e-4
