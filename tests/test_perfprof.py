"""
Recall that data[i, j] > 0 measures the performance of the
j-th solver on the i-th problem

To run tests:
python3 -m venv .test
python3 setup.py install --user
python3 -m pytest [-v]
"""
import numpy as np

from perfprof.perfprof import _best_timings
from perfprof.perfprof import _theta_max
from perfprof.perfprof import _theta
from perfprof.perfprof import _make_staircase

def test_best_timings():
    nice_data = np.array([[1., 2., 3.],
                          [6., 5., 4.]])
    
    assert np.array_equal(_best_timings(nice_data), [1., 4.])

    ugly_data = np.array([[np.nan, np.nan, np.nan],
                          [1., np.inf, np.inf]])
    
    assert np.array_equal(_best_timings(ugly_data), [np.inf, 1.])


def test_theta_max():
    nice_data = np.array([[1., 2., 3.],
                          [6., 5., 4.]])
    
    assert _theta_max(nice_data, _best_timings(nice_data)) == 3.

    ugly_data = np.array([[np.nan, np.nan, np.nan],
                          [1., np.inf, np.inf]])
    
    assert _theta_max(ugly_data, _best_timings(ugly_data)) == 1.01


def test_theta():
    nice_input = np.array([[1., 2., 3.],
                           [6., 5., 4.]])
    assert np.array_equal(_theta(nice_input[:, 0], _best_timings(nice_input)), [1., 1.5])
    assert np.array_equal(_theta(nice_input[:, 1], _best_timings(nice_input)), [2., 1.25])
    assert np.array_equal(_theta(nice_input[:, 2], _best_timings(nice_input)), [3., 1.])

    hard_input = np.array([[np.nan, np.nan, np.nan],
                           [1., np.inf, np.inf]])
    assert np.array_equal(_theta(hard_input[:, 0], _best_timings(hard_input)), [np.inf, 1.])
    assert np.array_equal(_theta(hard_input[:, 1], _best_timings(hard_input)), [np.inf, np.inf])
    assert np.array_equal(_theta(hard_input[:, 2], _best_timings(hard_input)), [np.inf, np.inf])


def test_make_staircase():
    # easy case
    easy_theta = np.array([2., 4., 3.])
    assert np.array_equal(_make_staircase(easy_theta[easy_theta <= 5.], len(easy_theta), 5., 0.1),
        [[1., 2., 3., 4., 5.], [0., 1/3, 2/3, 1., 1.]])
    assert np.array_equal(_make_staircase(easy_theta[easy_theta <= 4.], len(easy_theta), 4., 0.1),
        [[1., 2., 3., 4.], [0., 1/3, 2/3, 1.]])
    assert np.array_equal(_make_staircase(easy_theta[easy_theta <= 3.5], len(easy_theta), 3.5, 0.1),
        [[1., 2., 3., 3.5], [0., 1/3, 2/3, 2/3]])
    assert np.array_equal(_make_staircase(easy_theta[easy_theta <= 3.], len(easy_theta), 3., 0.1),
        [[1., 2., 3.], [0., 1/3, 2/3]])
    assert np.array_equal(_make_staircase(easy_theta[easy_theta <= 2.], len(easy_theta), 2., 0.1),
        [[1., 2.], [0., 1/3]])
    assert np.array_equal(_make_staircase(easy_theta[easy_theta <= 1.01], len(easy_theta), 1.01, 0.1),
        [[1., 1.01], [0., 0.]])
    assert np.array_equal(_make_staircase(easy_theta[easy_theta <= 1.01], len(easy_theta), 1.01, 0.001),
        [[1., 1.01], [0., 0.]])

    # easy case but with theta including 1.
    easy_theta1 = np.array([1., 3., 2.])
    assert np.array_equal(_make_staircase(easy_theta1[easy_theta1 <= 4.], len(easy_theta1), 4., 0.1),
        [[1., 2., 3., 4.], [1/3, 2/3, 1., 1.]])
    assert np.array_equal(_make_staircase(easy_theta1[easy_theta1 <= 3.], len(easy_theta1), 3., 0.1),
        [[1., 2., 3.], [1/3, 2/3, 1.]])
    assert np.array_equal(_make_staircase(easy_theta1[easy_theta1 <= 2.5], len(easy_theta1), 2.5, 0.1),
        [[1., 2., 2.5], [1/3, 2/3, 2/3]])
    assert np.array_equal(_make_staircase(easy_theta1[easy_theta1 <= 2.], len(easy_theta1), 2., 0.1),
        [[1., 2.], [1/3, 2/3]])
    assert np.array_equal(_make_staircase(easy_theta1[easy_theta1 <= 1.01], len(easy_theta1), 1.01, 0.1),
        [[1., 1.01], [1/3, 1/3]])
    assert np.array_equal(_make_staircase(easy_theta1[easy_theta1 <= 1.01], len(easy_theta1), 1.01, 0.001),
        [[1., 1.01], [1/3, 1/3]])

    # hard case with duplicates and inf; not including 1.
    hard_theta = np.array([np.inf, 2., np.inf, 2.])
    assert np.array_equal(_make_staircase(hard_theta[hard_theta <= 3.], len(hard_theta), 3., 0.1),
        [[1., 2., 3.], [0., 0.5, 0.5]])
    assert np.array_equal(_make_staircase(hard_theta[hard_theta <= 2.], len(hard_theta), 2., 0.1),
        [[1., 2.], [0., 0.5]])
    assert np.array_equal(_make_staircase(hard_theta[hard_theta <= 1.01], len(hard_theta), 1.01, 0.1),
        [[1., 1.01], [0., 0.]])
    assert np.array_equal(_make_staircase(hard_theta[hard_theta <= 1.01], len(hard_theta), 1.01, 0.001),
        [[1., 1.01], [0., 0.]])

    # hard case with duplicates and inf; including 1.
    hard_theta1 = np.array([np.inf, 2., 1., np.inf, 2.])
    assert np.array_equal(_make_staircase(hard_theta1[hard_theta1 <= 3.], len(hard_theta1), 3., 0.1),
        [[1., 2., 3.], [1/5, 3/5, 3/5]])
    assert np.array_equal(_make_staircase(hard_theta1[hard_theta1 <= 2.], len(hard_theta1), 2., 0.1),
        [[1., 2.], [1/5, 3/5]])
    assert np.array_equal(_make_staircase(hard_theta1[hard_theta1 <= 1.01], len(hard_theta1), 1.01, 0.1),
        [[1., 1.01], [1/5, 1/5]])
    assert np.array_equal(_make_staircase(hard_theta1[hard_theta1 <= 1.01], len(hard_theta1), 1.01, 0.001),
        [[1., 1.01], [1/5, 1/5]])
