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
import numpy as np
import matplotlib.pyplot as plt


def _best_timings(data):
    """Row-wise minima. NaN values are treated like +infinity."""
    return np.min(data, axis=1, initial=np.inf, where=~np.isnan(data))


def _theta_max(data, minvals):
    """Compute maximal performance ratio among all problems."""
    assert np.all(minvals > 0)
    tmax = np.max(data, axis=1, initial=0, where=(data < np.inf))
    thmax = np.max(tmax / minvals, initial=1.01)
    return thmax


def _theta(col, minvals):
    """
    Performance ratios for an individual solver against the vector of minimum values.
    Problems that are not solved by any algorithm have their ratios set to +Inf.
    """
    assert np.all(minvals > 0)
    th = np.full(np.shape(col), np.inf)
    valid = (minvals < np.inf)
    th[valid] = col[valid] / minvals[valid]
    return th


def _make_staircase(theta, m, thmax, tol):
    """
    Assemble staircase (x, y) pairs.
    theta : theta values of an individual solver
    m : number of problems
    thmax : maximum value of theta for endpoint clamping
    tol : theta tolerance for endpoint clamping
    """
    x, counts = np.unique(theta, return_counts=True)
    prob = np.cumsum(counts) / m

    # Ensure endpoints plotted correctly
    if len(x) == 0 or x[0] >= 1 + tol:
        x = np.append(1, x)
        prob = np.append(0, prob)
    if len(x) <= 1 or x[-1] < thmax - tol:
        x = np.append(x, thmax)
        prob = np.append(prob, prob[-1])

    return x, prob


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

    minvals = _best_timings(data)

    if np.any(minvals <= 0):
        raise ValueError("Data contains non-positive performance measurements")

    if thmax is None:
        thmax = _theta_max(data, minvals)

    def make_plot(solver):
        col = _theta(data[:, solver], minvals)  # performance ratio
        col = col[col <= thmax]  # crop and remove infs/NaNs

        if len(col) == 0:
            return None

        th, prob = _make_staircase(col, m, thmax, tol)

        # plot current line and disable frame clipping (to support y-intercept marking)
        result = plt.step(th, prob, linestyle[solver], where='post', **kwargs)
        result[0].set_clip_on(False)

        return result

    h = [make_plot(solver) for solver in range(n)]

    # set axis limits
    plt.xlim([1, thmax])
    plt.ylim([0, 1.01])
    plt.xlabel('performance ratio')
    plt.ylabel('problems solved')

    return thmax, h
