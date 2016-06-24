#!/usr/bin/env python
import os
import numpy as np
from scipy.ndimage import imread
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def loadAll():
    #numberOfData = 4
    numberOfData = 16479
    channels = 3
    size = 64
    img = np.empty([numberOfData, channels, size, size], dtype=np.float)
    y = np.empty([numberOfData, 1], dtype=np.int8)

    # Read in training data
    with open('train.csv', 'r') as f:
        it = 0
        for line in f:
            oneImg = imread("images/" + line.split(',')[0])
            oneImg = oneImg.reshape((channels, size, size))
            img[it] = oneImg
            y[it] = int(line.split(',')[1])
            it += 1

    numberOfVal = 3000
    valImg = np.empty([numberOfVal, channels, size, size], dtype=np.float)
    valY = np.empty([numberOfVal, 1], dtype=np.int8)

    # Read in validation data
    with open('valTrain.csv', 'r') as f:
        it = 0
        for line in f:
            oneImg = imread("images/" + line.split(',')[0])
            oneImg = oneImg.reshape((channels, size, size))
            valImg[it] = oneImg
            valY[it] = int(line.split(',')[1])
            it += 1

    numberOfTest = 2752
    testImg = np.empty([numberOfTest, channels, size, size], dtype=np.float)
    testY = np.zeros([numberOfTest, 1], dtype=np.int8)

    # Read in validation data
    with open('test.csv', 'r') as f:
        it = 0
        for line in f:
            oneImg = imread("testImages/" + line.strip())
            oneImg = oneImg.reshape((channels, size, size))
            testImg[it] = oneImg
            it += 1

    return img, y, valImg, valY, testImg, testY

def load_dataset():
    numberOfData = 10
    #numberOfData = 16479
    channels = 3
    size = 64
    img = np.empty([numberOfData, channels * size * size], dtype=np.float)
    y = np.empty([numberOfData, 1], dtype=np.int8)

    # Read in training data
    with open('strain.csv', 'r') as f:
        it = 0
        for line in f:
            oneImg = imread("images/" + line.split(',')[0])
            img[it] = oneImg.ravel()
            y[it] = int(line.split(',')[1])
            it += 1

    img /= 255
    return img, y

def load_test():
    numberOfData = 4
    #numberOfData = 2752
    channels = 3
    size = 64
    img = np.empty([numberOfData, channels * size * size], dtype=np.float)

    # Read in test data
    with open('stest.csv', 'r') as f:
        it = 0
        for line in f:
            oneImg = imread("testImages/" + line.strip())
            img[it] = oneImg.ravel()
            it += 1

    img /= 255
    return img

if __name__ == '__main__':
    X, y = load_dataset()
    testX = load_test()
    print("==Finish reading==\n")
    numberOfData = testX.shape[0]
    numberOfTestData = testX.shape[0]

    clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(criterion='entropy'), n_estimators=100)
    clf.fit(X, y.ravel()) 
    print("==Finish fitting==\n")
    predict = clf.predict(testX)

    with open('result.csv', 'w') as file:
        with open('stest.csv', 'r') as f:
            file.write("id,label\n")
            i = 0
            for line in f:
                file.write("%s,%d\n" %(line.strip(), predict[i]))
                i += 1
