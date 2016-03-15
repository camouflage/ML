import numpy as np

if __name__ == '__main__':
    # Read dataset
    # Ref: http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt
    # numpy V1.10

    data = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(0, 385), max_rows=20)
    reference = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=(385,), max_rows=20)

    # Set the first col to one.
    data[:,0] = 1

    theta = np.zeros((data.shape[1], 1))

    result = np.dot(data, theta)

    print(result)
    print(theta.shape)
    print(data)
    print(reference)

    