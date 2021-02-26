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


def theta(col, minvals):
    """
    Performance ratios for an individual solver against the vector of minimum values.
    Problems that are not solved by any algorithm have their ratios set to Inf.
    """
    assert np.all(minvals > 0)
    th = np.full(np.shape(col), np.inf)
    valid = (minvals < np.inf)
    th[valid] = col[valid] / minvals[valid]
    return th


def makeStaircase(col, m, thmax, tol):
    """
    Assemble staircase (x, y) pairs.
    col : "column" of theta values
    m : number of problems
    thmax : maximum value of theta for endpoint clamping
    tol : theta tolerance for endpoint clamping
    """
    theta, counts = np.unique(col, return_counts=True)
    prob = np.cumsum(counts) / m

    # Ensure endpoints plotted correctly
    if theta[0] >= 1 + tol:
        theta = np.append(1, theta)
        prob = np.append(0, prob)
    if theta[-1] < thmax - tol:
        theta = np.append(theta, thmax)
        prob = np.append(prob, prob[-1])

    return theta, prob


def perfprof(data, linestyle, thmax = None, tol = np.sqrt(np.finfo(np.double).eps), **kwargs):
    """
    Peformance profile for the input data.

    Parameters
    ----------
    data : Array of timings/errors to plot.
           M-by-N matrix where data[i, j] > 0 measures the performance of the
           j-th solver on the i-th problem, with smaller values denoting "better".
    
    linestyle : List of line specs, e.g., ['o-r', '-.g']

    thmax : Maximum value of theta shown on the x-axis.
            Defaults to max(tm, 1.01), where tm is the largest finite performance ratio.

    tol : Tolerance for endpoint clamping.
          Defaults to sqrt(eps), where eps is the double precision machine accuracy.

    **kwargs : Optional keyword args to be forwarded to matplotlib.
    
    Returns
    -------
    thmax : Maximum value of theta shown on the x-axis, as
            supplied by the user or computed by the function.
    
    h : array of Line2D handles of the individual plot lines.
    """

    data = np.asarray(data).astype(np.double)
    m, n = data.shape  # `m` problems, `n` solvers

    # Check input
    if len(linestyle) < n:
        raise ValueError("Number of line specs < number of solvers")

    # Row-wise minima. NaN values are treated like +infinity.
    minvals = np.min(data, axis=1, initial=np.inf, where=~np.isnan(data))

    if np.any(minvals <= 0):
        raise ValueError("Data contains non-positive performance measurements")

    if thmax is None:
        thmax = thetaMax(data, minvals)

    h = [None] * n
    for solver in range(n):  # for each solver
        col = theta(data[:, solver], minvals)  # performance ratio
        col = col[col <= thmax] # crop and remove infs/NaNs

        if len(col) == 0:
            continue

        th, prob = makeStaircase(col, m, thmax, tol)

        # plot current line and disable frame clipping (to support y-intercept marking)
        h[solver] = plt.step(th, prob, linestyle[solver], where='post', **kwargs)
        h[solver][0].set_clip_on(False)

    # set axis limits
    plt.xlim([1, thmax])
    plt.ylim([0, 1.01])

    return thmax, h
