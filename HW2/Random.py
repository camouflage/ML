import numpy as np
# Ref: https://docs.python.org/3/library/csv.html
import sys



if __name__ == '__main__':
    np.set_printoptions(edgeitems=5)

    numberOfTest = 220245
    #data = np.zeros((numberOfTest, 1))
    data = np.genfromtxt("RandomResult.csv", dtype=np.int8, delimiter=',', skip_header=1, usecols=(1)).reshape(numberOfTest, 1)

    numberOfChanges = 20000
    pos = np.random.randint(numberOfTest, size=numberOfChanges)
    for i in range(numberOfChanges):
        if np.random.random() >= 0.5337:
            data[pos[i]] = 1
        else:
            data[pos[i]] = 0
            
    print(np.count_nonzero(data))
    # Print result
    with open('RandomResult.csv', 'w') as file:
        file.write("id,label\n")
        for i in range(0, numberOfTest):
            file.write("%d,%d\n" %(i, data[i]))
