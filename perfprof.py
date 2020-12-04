"""
Python module for performance profiles. The code is based on
`perfprof` from the MATLAB Guide by D. J. Higham and N. J. Higham:
https://github.com/higham/matlab-guide-3ed/blob/master/perfprof.m

    References
    ----------
    [1] E. D. Dolan, and J. J. More,
        Benchmarking Optimization Software with Performance Profiles.
        Math. Programming, 91:201-213, 2002.
"""

__all__ = ['perfprof']

import numpy as np
import matplotlib.pyplot as plt


def thetaMax(data, minvals):
    """
    """
    assert np.all(minvals > 0)
    tmax = np.max(data, axis=1, initial=0, where=(data < np.inf))
    thmax = np.max(tmax / minvals, initial=1.01)
    return thmax


def thetaColumn(col, minvals):
    """
    Performance ratios for an individual solver against the vector of minimum values.
    Division by Inf produces NaN.
    """
    assert np.all(minvals > 0)
    th = np.full(np.shape(col), np.nan)
    th[minvals < np.inf] = col / minvals
    return th


def makeStaircase(col, thmax, tol):
    theta, counts = np.unique(col, return_counts=True)
    r = len(theta)
    prob = np.cumsum(counts) / len(col)    
    # TODO: get rid of floating arange
    k = np.array(np.floor(np.arange(0, r, 0.5)), dtype = np.int)
    x = theta[k[1:]]
    y = prob[k[:-1]]
    return x, y


def perfprof(data, thmax = None, tol = np.sqrt(np.finfo(np.double).eps)):
    """
    Peformance profile for the input data.

    Parameters
    ----------
    data : Array of timings/errors to plot.
           M-by-N matrix where data[i, j] > 0 measures the performance of the
           j-th solver on the i-th problem, with smaller values denoting "better".

    thmax : Maximum value of theta shown on the x-axis.
            Defaults to max(tm, 1.01), where tm is the largest finite performance ratio.

    tol : Tolerance for endpoint clamping.
          Defaults to sqrt(eps), where eps is the double precision machine accuracy.
    
    Returns
    -------
    thmax : Maximum value of theta shown on the x-axis, as
            supplied by the user or computed by the function.
    
    h : array of Line2D handles of the individual plot lines.
    """

    data = np.asarray(data).astype(np.double)
    m, n = data.shape  # `m` problems, `n` solvers

    # Row-wise minima. NaN values are treated like +infinity.
    minvals = np.min(data, axis=1, initial=np.inf, where=~np.isnan(data))

    # Check for invalid performance measurements
    if np.any(minvals <= 0):
        raise Exception("Data contains non-positive performance measurements")

    if thmax is None:
        thmax = thetaMax(data, minvals)

    h = [None] * n
    for solver in range(n):  # for each solver
        col = thetaColumn(data[:, solver], minvals)  # performance ratio
        col = col[col <= thmax] # crop and remove infs/NaNs

        x, y = makeStaircase(col, thmax, tol)

        # Ensure endpoints plotted correctly
        if x[0] >= 1 + tol:
            x = np.append([1, x[0]], x)
            y = np.append([0, 0], y)
        if x[-1] < thmax - tol:
            x = np.append(x, thmax)
            y = np.append(y, y[-1])

        # plot current line
        # TODO: use plt.step
        h[solver] = plt.plot(x, y)

    # set xlim
    plt.xlim([1, thmax])
    plt.ylim([0, 1.01])
    plt.show()  #plt.draw()?

    return thmax, h
