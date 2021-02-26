This is a Python script for printing *performance profiles* as defined by
[E. D. Dolan and J. J. Mor&eacute;](http://dx.doi.org/10.1007/s101070100263).

It is based on [`perfprof.m`](https://github.com/higham/matlab-guide-3ed/blob/master/perfprof.m)
from the MATLAB Guide by D. J. Higham and N. J. Higham.

## Usage

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

Legends can be displayed using `perprof.legend`, which is an alias for `matplotlib.legend`.

```python
import matplotlib.pyplot as plt
import perfprof

palette = ['o-r', 'o:b', 'o--c', 'o-.g', 'o-y']
legend = ['Algorithm 1', 'Algorithm 2']

perfprof.perfprof(data, palette, markersize=4, markevery=[0])
plt.legend(legend)
plt.show()
```
