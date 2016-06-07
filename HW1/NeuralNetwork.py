import numpy as np
from math import *
import sys

# Activation function: sigmoid
def sigmoid(m):
    return 1 / (1 + np.exp(-m))

def dSigmoid(m):
    #return sigmoid(m) * (1 - sigmoid(m))
    return m * (1 - m)


if __name__ == '__main__':
    # Read dataset
    data = np.genfromtxt("shuffled.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    reference = np.genfromtxt("shuffled.csv", delimiter=',', skip_header=1, usecols=(385))

    m = data.shape[0]
    n = data.shape[1]
    reference = reference.reshape(m, 1)
    y = reference
    
    # Normalization
    maxNumber = y.max()
    minNumber = y.min()
    y = (y - minNumber) / (maxNumber - minNumber)

    # Init theta, alpha, number of iterations, number of neurons in the hidden layer
    numberOfNeurons = 120

    weight0 = np.genfromtxt("weight0", delimiter=' ').reshape(n, numberOfNeurons)
    weight1 = np.genfromtxt("weight1", delimiter=' ').reshape(numberOfNeurons, 1)
    hiddenTheta = np.genfromtxt("hiddenTheta", delimiter=' ').reshape(1, numberOfNeurons)
    outputTheta = np.genfromtxt("outputTheta", delimiter=' ').reshape(1, 1)

    #weight0 = 0.01 * np.random.randn(n, numberOfNeurons)
    #weight1 = 0.01 * np.random.randn(numberOfNeurons, 1)
    #hiddenTheta = np.zeros((1, numberOfNeurons))
    #outputTheta = 0
    #hiddenTheta = np.random.rand(1, numberOfNeurons)
    #outputTheta = np.random.rand()


    alpha = 0.4
    iteration = 250
    lastJ = sys.maxsize

    for i in range(0, iteration):
        # Stochastic gradient descent
        for row in range(0, m):
            chosen = data[row]

            hiddenIn = np.dot(chosen, weight0) + hiddenTheta
            hiddenOut = sigmoid(hiddenIn)
            #print(hiddenIn, hiddenOut)

            outputIn = np.dot(hiddenOut, weight1) + outputTheta
            outputOut = sigmoid(outputIn)
            #print(outputIn, outputOut)

            # Calculate error
            outputError = dSigmoid(outputOut) * (y[row] - outputOut)
            hiddenError = dSigmoid(hiddenOut) * outputError * np.transpose(weight1)
            #print(outputError)
            #print(hiddenError)

            # Update output layer theta and weight
            outputTheta += alpha * outputError
            weight1 += alpha * outputError * hiddenOut.reshape(numberOfNeurons, 1)

            # Update hidden layer theta and weight
            hiddenTheta += alpha * hiddenError
            weight0 += alpha * np.outer(data[row], hiddenError)

            #print(weight0)
            #print(weight1)
            #print(hiddenTheta)
            #print(outputTheta)

        hidden = sigmoid(np.dot(data, weight0) + hiddenTheta)
        output = sigmoid(np.dot(hidden, weight1) + outputTheta)
        output = output * (maxNumber - minNumber) + minNumber
        J = np.sum(np.square(output - reference)) / (m * 2)

        if J > lastJ:
            alpha = alpha / 2
            print("Alpha changed to %f at iteration %d"  %(alpha, i))
        if alpha < 0.02:
            break

        lastJ = J
        print(i, J)


    np.savetxt("weight0", weight0, fmt="%f", delimiter=' ')
    np.savetxt("weight1", weight1, fmt="%f", delimiter=' ')
    np.savetxt("hiddenTheta", hiddenTheta, fmt="%f", delimiter=' ')
    np.savetxt("outputTheta", outputTheta, fmt="%f", delimiter=' ')

    testData = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    testM = testData.shape[0]

    testHidden = sigmoid(np.dot(testData, weight0) + hiddenTheta)
    testOutput = sigmoid(np.dot(testHidden, weight1) + outputTheta)
    testOutput = testOutput * (maxNumber - minNumber) + minNumber

    with open('ans.csv', 'w') as file:
        file.write("id,reference\n")
        for i in range(0, testM):
            file.write("%d,%f\n" %(i, testOutput[i]))
