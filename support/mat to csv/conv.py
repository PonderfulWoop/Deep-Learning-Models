import scipy.io
import numpy as np

for i in range(1, 183):
	data = scipy.io.loadmat("GT_IMG_"+str(n)+".mat")

for i in data:
	if '__' not in i and 'readme' not in i: # tis a double underscore
		np.savetxt((i+".csv"), data[i], fmt = '%s', delimiter = ',')
