import numpy as np
import matplotlib.pyplot as plt
import perfprof


# Load up some data to show. This is taken from actual research!
measurements = np.loadtxt('example.csv', delimiter=',')

# Choose the line styles we'd like to use. Note how there are 7 line styles
# for 4 algorithms, but that's OK. It makes the code easier to extend.
palette = ['o-C0', 'o:C1', 'o--C2', 'o-.C3', 'o-C4', 'o:C5', 'o-C6']

# Let's get to work! The 'markersize' and 'markevery' arguments help us
# emphasise y-intercepts in the plot, which have special meaning.
perfprof.perfprof(measurements, linestyle=palette, thmax=5., markersize=4, markevery=[0])

# We can use standard matplotlib commands to work on the output figure
plt.legend(['Alg1', 'Alg2', 'Alg3', 'Alg4'], loc=4, fontsize=16)
plt.show()
