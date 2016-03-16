import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Read dataset
    # Ref: http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt
    # numpy V1.10

    #data = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(0, 385))
    #reference = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=(385))
    data = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(0, 385), max_rows=20000)
    reference = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=(385), max_rows=20000)

    m = data.shape[0]
    n = data.shape[1]
    y = reference.reshape(m, 1)

    # Set the first col to one.
    data[:,0] = 1
    # Init theta, alpha, number of iteration
    theta = np.zeros((n, 1))
    alpha = 0.03
    iteration = 500
    JList = []

    for i in range(0, iteration):
        hTheta = np.dot(data, theta)
        hThetaSubY = np.subtract(hTheta, y)

        J = np.sum(np.square(hThetaSubY)) / (m * 2)
        JList.append(J)

        hThetaMulX = np.multiply(hThetaSubY, data)
        # Ref: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sum.html
        sigma = np.sum(hThetaMulX, axis=0).reshape(n, 1)
        theta = np.subtract(theta, np.multiply(sigma, alpha / m))

    plt.plot(JList)
    plt.show()

    with open('theta', 'w') as file:
        for i in theta:
            file.write(str(i))

    testData = np.genfromtxt("train.csv", delimiter=',', skip_header=20001, usecols=range(0, 385))
    testReference = np.genfromtxt("train.csv", delimiter=',', skip_header=20001, usecols=(385))
    testData[:,0] = 1
    testM = testData.shape[0]
    testY = testReference.reshape(testM, 1)
    
    testHTheta = np.dot(testData, theta)
    testHThetaSubY = np.subtract(testHTheta, testY)
    with open('result', 'w') as file:
        for i in range(0, testM):
            file.write("%f  %f  %f\n" %(testHTheta[i], testReference[i], testHThetaSubY[i]))
        file.write(str(np.sum(np.square(hThetaSubY)) / m))

    """
    testData = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(0, 385))
    testData[:,0] = 1

    testHTheta = np.dot(testData, theta)
    testM = testData.shape[0]
    with open('ans', 'w') as file:
        file.write("id,reference\n")
        for i in range(0, testM):
            file.write("%d,%f\n" %(i, testHTheta[i]))
    """