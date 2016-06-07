import numpy as np
import sys
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor


if __name__ == '__main__':
    np.set_printoptions(edgeitems=5)

    # Read dataset
    data = np.genfromtxt("shuffled.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    reference = np.genfromtxt("shuffled.csv", delimiter=',', skip_header=1, usecols=(385))
    testData = np.genfromtxt("test.csv", delimiter=',', skip_header=1, usecols=range(1, 385))
    validationData = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=range(1, 385), max_rows=5000)
    validationReference = np.genfromtxt("train.csv", delimiter=',', skip_header=1, usecols=(385), max_rows=5000)

    numberOfTrainingData = data.shape[0]
    numberOfFeatures = data.shape[1]
    numberOfTestData = testData.shape[0]
    numberOfVldtData = validationData.shape[0]

    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    bdt = AdaBoostRegressor(base_estimator=ExtraTreeRegressor(), n_estimators=1000)
    #bdt = RandomForestRegressor(n_estimators=50)
    #bdt = GradientBoostingRegressor()

    bdt.fit(data, reference)
    print("FINISH FITTING")
    predict = bdt.predict(testData).reshape(numberOfTestData, 1)
    score = bdt.score(validationData, validationReference)
    print(score)

    with open('adaboostResult.csv', 'w') as file:
        file.write("id,reference\n")
        for i in range(0, numberOfTestData):
            file.write("%d,%f\n" %(i, predict[i]))
