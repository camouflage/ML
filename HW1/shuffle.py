import numpy as np
from math import *

if __name__ == '__main__':
    # Read dataset
    # Ref: http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt
    # numpy V1.10

    data = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(0, 386))
    np.random.shuffle(data)

    np.savetxt("shuffled.csv", data, fmt="%.6f", header="dummy", delimiter=',')
    
