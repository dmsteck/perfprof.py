This is a Python script for printing *performance profiles* as defined by
[E. D. Dolan and J. J. Mor&eacute;](http://dx.doi.org/10.1007/s101070100263).

It is based on [`perfprof.m`](https://github.com/higham/matlab-guide-3ed/blob/master/perfprof.m)
from the MATLAB Guide by D. J. Higham and N. J. Higham.

## Performance profiles

Performance profiles are a mechanism to visualise the performance of multiple algorithms on multiple test problems.
Given `m` problems and `n` algorithms, we're interested in the relative performance of all algorithms across the entire problem set.

Let `t = t(i, j) > 0` be a measure of the performance of solver `j` on problem `i`, where lower means "better".
Common choices for `t` are:

*  execution/CPU time;
*  number of iterations (assuming the algorithms are iterative);
*  the number of evaluations of some reference function (e.g., the ordering predicate in a sorting algorithm, or the objective function in an optimisation algorithm).

Given this data, a typical performance profile may look like this:

![Example](https://raw.githubusercontent.com/dmsteck/perfprof.py/master/examples/example.svg "Example performance profile")

Each algorithm has one line plot, where a point `(x, y)` means that, for `x` of the problem set, the algorithm in question was within a factor of `y` of the respective best algorithm.
For example:

*  the point &asymp;`(1, 0.6)` means that `Alg2` was the fastest algorithm on around 60% of the problem set;
*  the point &asymp;`(1.5, 0.4)` means that `Alg3` was within a factor of 1.5 of the respective best algorithm for 40% of problems;
   *  note that the "best" algorithm may be different for each problem;
*  the point &asymp;`(1.5, 0.95)` means that `Alg1` was within a factor of 1.5 of the respective best algorithm for 95% of problems;
*  etc.

Generally speaking, an algorithm is considered efficient (relative to the others) when its performance profile comes close to the top left corner `(1, 1)`.

It is possible for algorithms to fail on certain problems.
This can be achieved by simply setting the performance measure `t(i, j)` to `+inf` or `NaN`.

## Usage examples

```python
import matplotlib.pyplot as plt
import perfprof

palette = ['-r', ':b', '--c', '-.g', '-y']

perfprof.perfprof(data, palette)
plt.show()
```

### Marking y-intercepts

Markers can be inserted using the standard `matplotlib` pattern.

```python
import matplotlib.pyplot as plt
import perfprof

palette = ['o-r', 'o:b', 'o--c', 'o-.g', 'o-y']

perfprof.perfprof(data, palette, markersize=4, markevery=[0])
plt.show()
```

### Displaying legends

Legends can be displayed using `matplotlib.pyplot.legend`.

```python
import matplotlib.pyplot as plt
import perfprof

palette = ['o-r', 'o:b', 'o--c', 'o-.g', 'o-y']
legend = ['Algorithm 1', 'Algorithm 2']

perfprof.perfprof(data, palette, markersize=4, markevery=[0])
plt.legend(legend)
plt.show()
```

## Why another implementation?

Multiple implementations of performance profiles already exist in the public domain.

The design of `perfprof` was driven by a few key desires:

*  **Simplicity:** provide a clearly scoped, easy to use implementation that integrates with `matplotlib`;
*  **Flexibility:** unlock the full power of `matplotlib` for plot styling, legends, subplots etc.;
*  **Robustness:** the implementation must work in all edge cases including `inf`, `NaN`, etc.;
*  **Usability:** full Python3 compatibility and sensible defaults where possible.
