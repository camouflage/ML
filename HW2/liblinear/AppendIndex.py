import numpy as np
# Ref: https://docs.python.org/3/library/csv.html
import sys

if __name__ == '__main__':
    np.set_printoptions(edgeitems=5)

    numberOfTest = 220245
    data = np.genfromtxt("output.txt", dtype=np.int8).reshape(numberOfTest, 1)

    # Print result
    with open('appendedOutput.csv', 'w') as file:
        file.write("id,label\n")
        for i in range(0, numberOfTest):
            file.write("%d,%d\n" %(i, data[i]))
