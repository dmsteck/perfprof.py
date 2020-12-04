import numpy as np
import perfprof


#data = np.array([[1, 4, 7],
#                 [5, 8, 2],
#                 [9, 3, 6]])

data = np.array([[1, 2], [np.inf, 3], [1, 1]])

perfprof.perfprof(data)

#print(data)
