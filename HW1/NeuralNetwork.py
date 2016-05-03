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
    data = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    reference = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=(385))
    #data = np.genfromtxt("shuffled.csv", delimiter=',', skip_header=1, usecols=range(1, 6), max_rows=7)
    #reference = np.genfromtxt("shuffled.csv", delimiter=',', skip_header=1, usecols=(385), max_rows=7)
    #data = np.genfromtxt("toy.csv", delimiter=',', skip_header=1, usecols=range(0, 2))
    #reference = np.genfromtxt("toy.csv", delimiter=',', skip_header=1, usecols=(2))

    m = data.shape[0]
    n = data.shape[1]
    reference = reference.reshape(m, 1)
    y = reference
    
    # Normalization
    maxNumber = y.max()
    minNumber = y.min()
    y = (y - minNumber) / (maxNumber - minNumber)

    # Init theta, alpha, number of iterations, number of neurons in the hidden layer
    numberOfNeurons = ceil(sqrt(n))
    #weight0 = np.ones((n, numberOfNeurons)) * 0.1
    #weight1 = np.ones((numberOfNeurons, 1)) * 0.1
    #hiddenTheta = np.ones((1, numberOfNeurons)) * 0.2
    #outputTheta = 0.2

    # Weight0 with theta included
    weight0 = np.random.rand(n, numberOfNeurons)
    weight1 = np.random.rand(numberOfNeurons, 1)
    hiddenTheta = np.random.rand(1, numberOfNeurons)
    outputTheta = np.random.rand()
    alpha = 0.6
    iteration = 200

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
            hiddenError = dSigmoid(outputOut) * outputError * weight1
            #print(outputError)
            #print(hiddenError)

            # Update output layer theta and weight
            outputTheta += alpha * outputError
            weight1 += alpha * outputError * hiddenOut.reshape(numberOfNeurons, 1)

            # Update hidden layer theta and weight
            hiddenTheta += alpha * np.transpose(hiddenError)
            weight0 += np.outer(data[row], hiddenError)

            #print(weight0)
            #print(weight1)
            #print(hiddenTheta)
            #print(outputTheta)

        hidden = sigmoid(np.dot(data, weight0) + hiddenTheta)
        output = sigmoid(np.dot(hidden, weight1) + outputTheta)
        output = output * (maxNumber - minNumber) + minNumber
        J = np.sum(np.square(output - reference)) / (m * 2)
        print(i, J)

    testData = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    #testData = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(1, 6), max_rows=7)
    testM = testData.shape[0]

    testHidden = sigmoid(np.dot(testData, weight0) + hiddenTheta)
    testOutput = sigmoid(np.dot(testHidden, weight1) + outputTheta)
    testOutput = testOutput * (maxNumber - minNumber) + minNumber

    with open('ans', 'w') as file:
        file.write("id,reference\n")
        for i in range(0, testM):
            file.write("%d,%f\n" %(i, testOutput[i]))
