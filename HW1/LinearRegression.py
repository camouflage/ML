import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Read dataset
    # Ref: http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt
    # numpy V1.10

    #index = np.genfromtxt("trainIndex.csv", dtype=int)
    data = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(0, 385))
    #data = data[np.array(index)]
    reference = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=(385))
    #reference = reference[np.array(index)]

    #data = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(0, 385), max_rows=20000)
    #reference = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=(385), max_rows=20000)

    m = data.shape[0]
    n = data.shape[1]
    y = reference.reshape(m, 1)

    # Set the first col to one.
    data[:,0] = 1
    # Init theta, alpha, number of iteration
    #theta = np.genfromtxt("theta full").reshape(n, 1)
    theta = np.zeros((n, 1))
    squareTheta = np.zeros((n, 1))
    alpha = 0.075
    #myLambda = 10
    iteration = 15000

    for i in range(0, iteration):
        hTheta = np.dot(data, theta)
        hThetaSubY = np.subtract(hTheta, y)

        J = np.sum(np.square(hThetaSubY)) / (m * 2)
        #np.square(theta, squareTheta)
        #J = (np.sum(np.square(hThetaSubY)) + myLambda * np.sum(squareTheta)) / (m * 2)
        if i % 50 == 0:
            print(i, J)

        hThetaMulX = np.multiply(hThetaSubY, data)
        # Ref: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sum.html
        sigma = np.sum(hThetaMulX, axis=0).reshape(n, 1)
        #theta = np.multiply(theta, 1 - alpha * myLambda / m)
        theta = np.subtract(theta, np.multiply(sigma, alpha / m))

    #plt.plot(JList)
    #plt.show()

    with open('theta', 'w') as file:
        for i in theta:
            file.write("%f\n" %(i[0]))

    """
    # Second part of training set as testing.
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
    # Filter out good samples
    testHTheta = np.dot(data, theta)
    testHThetaSubY = np.subtract(testHTheta, y)
    with open('trainIndex.csv', 'w') as file:
        for i in range(0, m):
            if abs(testHThetaSubY[i]) < 30:
                file.write("%d\n" %(i))
            #file.write("%f  %f  %f\n" %(testHTheta[i], y[i], testHThetaSubY[i]))
        #file.write(str(np.sum(np.square(hThetaSubY)) / m))
    """