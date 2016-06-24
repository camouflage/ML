from __future__ import division
import numpy as np
import itertools
import csv
import math  
import sys
import threading

from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import load_iris  


class Samples:
    def __init__(self, training_data=None):
        self.training_data = training_data

    def generateSampleId(self):
        sample_id = []
        for i in range(len(self.training_data)):
            sample_id.append(np.random.randint(len(self.training_data)))
        return sample_id


class node:
    def __init__(self, col=-1, value=None, results=None, trueBranch=None, falseBranch=None):  
        self.col = col  
        self.value = value  
        self.results = results  
        self.trueBranch = trueBranch  
        self.falseBranch = falseBranch  
          
    def getLabel(self):  
        if self.results == None:  
            return None  
        else:  
            max_counts = 0  
            for key in self.results.keys():  
                if self.results[key] > max_counts:  
                    label = key  
                    max_counts = self.results[key]

        return label


class RandomForestsClassifier:  
    def __init__(self, sample, n_bootstrapSamples=20):  
        self.n_bootstrapSamples = n_bootstrapSamples  
        self.trainingSample = Samples(sample)
        self.data = sample
        self.list_tree = []

    def training(self):
        sample_id = self.trainingSample.generateSampleId()
        currentTree = self.buildTree(sample_id)
        self.list_tree.append(currentTree)

    def multithreadTraining(self):
        numberOfThreads = 2
        for i in range(self.n_bootstrapSamples / numberOfThreads):
            for i in range(numberOfThreads):
                t = threading.Thread(target=training)
                t.start()

            for i in range(numberOfThreads):
                t.join()

    def buildTree(self, sample_data):
        if len(sample_data) == 0:
            return node()
        
        currentGini = self.giniEstimate(sample_data)
        bestGain = 0
        bestDivide = None
        bestCriteria = None

        colCount = len(self.data[sample_data[0]])
        colRange = [i for i in range(1, colCount)]
        num_TrainingAttr = int(math.ceil(math.sqrt(colCount)))

        np.random.shuffle(colRange)

        for col in colRange[0:num_TrainingAttr]:
            (trueSet, falseSet) = self.divideSet(sample_data, col)
            gain = currentGini - (len(trueSet) * self.giniEstimate(trueSet)
                + len(falseSet) * self.giniEstimate(falseSet))
            if gain > bestGain and len(trueSet) > 0 and len(falseSet) > 0:
                bestGain = gain
                bestCriteria = (col, 1)
                bestDivide = (trueSet, falseSet)

        if bestGain > 0:
            trueBranch = self.buildTree(bestDivide[0])
            falseBranch = self.buildTree(bestDivide[1])
            return node(col=bestCriteria[0], value=bestCriteria[1], trueBranch=trueBranch, falseBranch=falseBranch)
        else:
            return node(results=self.differentResult(sample_data))

    def giniEstimate(self, sample_data):
        if len(sample_data) == 0:
            return 0

        total_row = len(sample_data)
        different_result = self.differentResult(sample_data)
        gini = 0
        for i in different_result:
            gini = gini + pow(different_result[i], 2)
        gini = 1 - gini / pow(total_row, 2)
        return gini

    def differentResult(self, sample_data):
        results = {}
        for row in sample_data:
            value = (self.data)[row][0]
            if value not in results:
                results[value] = 0
            results[value] += 1
        return results

    def divideSet(self, sample_data, col):
        set1 = [row for row in sample_data if int((self.data)[row][col]) == 1]
        set2 = [row for row in sample_data if int((self.data)[row][col]) == 0]
        return (set1, set2)

    def predict(self, test_data):
        results = {}
        for i in range(len(self.list_tree)):
            currentResult = self.predict_tree(test_data, self.list_tree[i])
            if currentResult not in results:
                results[currentResult] = 0
            results[currentResult] += 1

        max_counts = 0
        for key in results.keys():
            if results[key] > max_counts:
                finalResult = key
                max_counts = results[key]

        return finalResult

    def predict_tree(self, testdata, tree):
        if tree.results != None:
            return tree.getLabel()
        else:
            currentValue = testdata[tree.col]
            branch = None
            if currentValue == tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch

        return self.predict_tree(testdata, branch)


if __name__ == '__main__':
    start = 0
    #end = 15000
    end = 150

    rows = end - start
    cols = 11392 + 1

    # Read in reference
    reference = np.genfromtxt("train.txt", dtype=np.int8, usecols=(0), skip_header=start, max_rows=rows)

    X = []
    with open("train.txt") as trainingFile:
        reader = csv.reader(itertools.islice(trainingFile, start, end), delimiter=' ')
        
        index = 0
        for row in reader:
            oneRow = [0] * cols
            oneRow[0] = reference[index]

            elementList = [ele.split(':') for ele in row[1:]]
            for ele in elementList:
                oneRow[int(ele[0])] = 1
            
            X.append(oneRow)
            index += 1
    
    testX = []
    testRow = 4
    with open("stest.txt") as testFile:
        reader = csv.reader(itertools.islice(testFile, 0, testRow), delimiter=' ')
        
        index = 0
        for row in reader:
            oneRow = [0] * cols
            oneRow[0] = index

            elementList = [ele.split(':') for ele in row[1:]]
            for ele in elementList:
                oneRow[int(ele[0])] = 1
            
            testX.append(oneRow)
            index += 1


    mid = int(end * 0.7)
    lastmid = end * 0.3
    rowRange = [i for i in range(end)]

    np.random.shuffle(rowRange) 
    # Randomly choose 70% data 
    trainX = [X[i] for i in rowRange[0:mid]]  
    # Randomly choose 30% data as cross reference  
    checkX = [X[i] for i in rowRange[mid:end]]  


    classifier = RandomForestsClassifier(trainX, n_bootstrapSamples=150)
    classifier.training()
    finalResults = []
    for row in checkX:
        finalResult = classifier.predict(row)
        finalResults.append(finalResult)

    errorResult = np.zeros((lastmid,1))
    errorResult[np.array(finalResults) != (np.array(checkX))[:,0]] = 1
    
    print("Error Rate: %f" %(errorResult.sum() / lastmid))
