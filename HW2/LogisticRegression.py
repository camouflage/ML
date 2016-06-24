# Created By Sheng Sun
import itertools
import csv
import sys
import threading

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from scipy.special import expit
#from concurrent.futures import *
#from multiprocessing.dummy import Pool as ThreadPool

# Train
def train(partialX, partialY, theta):
    """
    Using vectorization which speeds up a lot
    """
    # Sigmoid Ref: http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html
    h = expit(partialX.dot(theta))
    J = np.sum(np.multiply(partialY, np.log(h)) + np.multiply(1 - partialY, np.log(1 - h))) / -batchSize 
    print("At iteration %d: J = %f\n" %(it, J))
    partialDerivativeJ = np.sum(partialX.multiply(h - partialY), axis=0).reshape(cols, 1)
    theta = theta - alpha / rows * partialDerivativeJ

# Train in two threads
def multiThreadTrain(partialX, partialY, theta, st, end):
    """
    Multithread version.
    Ref: https://pymotw.com/2/threading/
    """
    h = expit(partialX.dot(theta))
    partialJ = np.sum(np.multiply(partialY, np.log(h)) + np.multiply(1 - partialY, np.log(1 - h))) / -batchSize 
    print("At iteration %d: partialJ = %f" %(it, partialJ))
    partialDerivativeJ = np.sum(np.multiply(partialX, h - partialY), axis=0).reshape(cols, 1)
    theta[st:end] = theta[st:end] - alpha / rows * partialDerivativeJ[st:end]


# Predict
def predict(theta):
    if mode == "DEV":
        testFile = load_svmlight_file("strain.txt", n_features=cols, dtype=np.int8)
        realTestFile = load_svmlight_file("strain.txt", n_features=cols, dtype=np.int8)
    elif mode == "TEST":
        testFile = load_svmlight_file("testTest.txt", n_features=cols, dtype=np.int8)
        realTestFile = load_svmlight_file("testTest.txt", n_features=cols, dtype=np.int8)
    elif mode == "TRAIN":
        testFile = load_svmlight_file("testTest.txt", n_features=cols, dtype=np.int8)
        realTestFile = load_svmlight_file("test.txt", n_features=cols, dtype=np.int8)
        
    # Calculate and print cross validation result
    testData = testFile[0]
    testRows = testData.shape[0]
    testRef = testFile[1].reshape(testRows, 1)
    testH = expit(testData.dot(theta))
    testY = np.greater(testH, 0.5)

    correctCount = np.count_nonzero(testRef == testY)
    
    with open('testResult.csv', 'w') as file:
        file.write("Correct Rate: %f\n" %(correctCount / testRows))
        file.write("id,real,predict\n")
        for i in range(0, testRows):
            file.write("%d,%d,%d\n" %(i, testRef[i], testY[i]))

    print("=====Finish cross validation=====")

    # Calculate test reuslt
    realTestData = realTestFile[0]
    realTestRows = realTestData.shape[0]
    realTestH = expit(realTestData.dot(theta))
    realTestY = np.greater(realTestH, 0.5)

    # Print test result
    with open('result.csv', 'w') as file:
        file.write("id,label\n")
        for i in range(0, realTestRows):
            file.write("%d,%d\n" %(i, realTestY[i]))

    print("=====Finish prediction=====") 


# main
if __name__ == '__main__':
    np.set_printoptions(edgeitems=5)

    # Set three modes for convenience in development 
    mode = "DEV"

    if mode == "DEV":
        rows = 5
        batchSize = 5
        iteration = 3
        start = 0
    elif mode == "TEST":
        rows = 57627
        batchSize = 6403
        iteration = 200
        start = 0
    elif mode == "TRAIN":
        rows = 2177020
        batchSize = 6403
        iteration = 10
        start = 0

    # One extra column for theta0 and b0
    cols = 11392 + 1

    # Learning rate
    alpha = 0.5

    """
    # Read in all the training data, but this is not feasible
    # Ref: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html
    trainingFile = load_svmlight_file("train.txt", n_features=cols, dtype=np.int8)
    data = trainingFile[0]
    reference = trainingFile[1].reshape(rows, 1)
    print(reference.shape)
    """    

    # Read in reference
    reference = np.genfromtxt("strain.txt", dtype=np.int8, usecols=(0), max_rows=rows)
    y = reference.reshape(rows, 1)

    # Initialize theta or read in from theta.csv
    #theta = np.zeros((cols, 1))
    theta = np.genfromtxt("theta.csv", delimiter=' ').reshape(cols, 1)

    # Read in data
    for curRow in range(start, rows, batchSize):
        print("At batch %d\n" %(curRow))
        with open("strain.txt") as trainingFile:
            # Ref: http://stackoverflow.com/questions/19031423/how-to-loop-through-specific-range-of-rows-with-python-csv-reader
            reader = csv.reader(itertools.islice(trainingFile, curRow, curRow + batchSize), delimiter=' ')
            
            # Init data, colIdx, rowIdx
            rowNumber = 0
            data = []
            colIdx = []
            rowIdx = []

            for row in reader:
                elementList = [ele.split(':') for ele in row[1:]]
                # Add one col as x0
                #subData = [1] + [int(ele[1]) for ele in elementList]
                subData = [1] * (len(elementList) + 1)
                #print("Row %d: %d entries are 1" %(rowNumber, len(subData)))
                subColIdx = [0] + [int(ele[0]) for ele in elementList]
                subRowIdx = [rowNumber] * (len(elementList) + 1)

                rowNumber += 1
                # Extend the main list
                data.extend(subData)
                colIdx.extend(subColIdx)
                rowIdx.extend(subRowIdx)

            # Ref: http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix
            partialX = csr_matrix((data, (rowIdx, colIdx)), shape=(batchSize, cols))
            partialY = y[curRow:curRow + batchSize]

            dividingPoint = int(batchSize / 2)
            firstPartX = partialX[:dividingPoint].toarray()
            secondPartX = partialX[dividingPoint:].toarray()
        
        for it in range(iteration):
            #train(partialX, partialY, theta)
            # Multithread version
            t0 = threading.Thread(target=multiThreadTrain,
                args=(firstPartX, partialY[:dividingPoint], theta, 0, dividingPoint))
            t1 = threading.Thread(target=multiThreadTrain,
                args=(secondPartX, partialY[dividingPoint:], theta, dividingPoint, -1))

            t0.start()
            t1.start()

            t0.join()
            t1.join()

            #executor = ThreadPoolExecutor(max_workers=2)
            #executor.submit(multiThreadTrain)

    print("=====Finish training=====")


    # Save theta
    with open('theta.csv', 'w') as file:
        for i in theta:
            file.write("%f\n" %(i[0]))

    predict(theta)
