"""
Python module for performance profiles. The code is based on
`perfprof` from the MATLAB Guide by D. J. Higham and N. J. Higham:
https://github.com/higham/matlab-guide-3ed/blob/master/perfprof.m

    References
    ----------
    [1] E.D. Dolan, and J. J. More,
        Benchmarking Optimization Software with Performance Profiles.
        Math. Programming, 91:201-213, 2002.
"""

__all__ = ['perfprof']

import math
import numpy as np
import matplotlib.pyplot as plt


def perfprof(data, thmax = None, tol = math.sqrt(np.finfo(np.double).eps)):
    """
    Peformance profile for the input data.

    Parameters
    ----------
    data : Array of timings/errors to plot.
           M-by-N matrix where data[i, j] measures the performance
           of the j-th solver on the i-th problem, with smaller
           values denoting "better".

    thmax : Maximum value of theta shown on the x-axis.
            If None then thmax defaults to the largest finite
            performance ratio, with a minimum value of 1.01.

    tol : Tolerance for endpoint clamping. Defaults to sqrt(eps),
          where eps is the double precision machine accuracy.
    
    Returns
    -------
    thmax : Maximum value of theta shown on the x-axis, as
            supplied by the user or computed by the function.
    
    h : array of Line2D handles of the individual plot lines.

    """

    data = np.asarray(data).astype(np.double)

    minvals = np.min(data, axis=1, initial=0, where=~np.isnan(data))

    # Discard invalid problem data
    valid = (minvals > 0)
    if ~np.any(valid):
        raise Exception("No valid problems in the dataset")
    data = data[valid, :]
    minvals = minvals[valid]

    if thmax is None:
        thmax = np.max(np.max(data, axis=1, initial=0, where=(data < np.inf)) / minvals)
        thmax = np.maximum(thmax, 1.01)

    m, n = data.shape  # `m` problems, `n` solvers
    h = [None] * n
    for solver in range(n):  # for each solver
        col = data[:, solver] / minvals  # performance ratio
        col = col[col <= thmax] # crop and remove infs/NaNs

        theta = np.unique(col)
        r = len(theta)

        # TODO: simplify
        myarray = np.repeat(col, r).reshape(len(col), r) <= \
            np.repeat(theta, len(col)).reshape((len(col), r), order='F')
        myarray = np.array(myarray, dtype=np.double)
        prob = np.sum(myarray, axis=0) / m

        # Get points to print staircase plot
        # TODO: get rid of floating arange
        k = np.array(np.floor(np.arange(0, r, 0.5)), dtype = np.int)
        x = theta[k[1:]]
        y = prob[k[0:-1]]

        # Ensure endpoints plotted correctly
        if x[0] >= 1 + tol:
            x = np.append([1, x[0]], x)
            y = np.append([0, 0], y)
        if x[-1] < thmax - tol:
            x = np.append(x, thmax)
            y = np.append(y, y[-1])

        # plot current line
        h[solver] = plt.plot(x, y)

    # set xlim
    plt.xlim([1, thmax])
    plt.ylim([0, 1.01])
#    plt.draw()
    plt.show()

    return thmax, h
