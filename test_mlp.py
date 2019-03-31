import mlp
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from random import shuffle
import numpy as np

def compare(results, targets):
    assert(len(results) == len(targets))
    numCorrect = 0;
    for r, t in zip(results, targets):
        if r == t:
            numCorrect += 1
    return numCorrect / len(results)

def parseCSV(filename):
    """Parse a CSV file into an ndarray"""
    file = open(filename)
    array = []
    for line in file:
        #the [:-1] is to ignore the newline at the end of each line
        array.append(line[:-1].split(','))
    file.close()
    return np.array(array)

print("XOR:")
clsfyr = mlp.train(2, [[0,0],[0,1],[1,0],[1,1]], [0,1,1,0])
results = mlp.classify(clsfyr, [[1,1],[0,0],[1,0],[0,1],[0,0],[0,1],[1,0],[1,1]])
print("accuracy: {0:.2f}%\n".format(compare(results, [0,0,1,1,0,1,1,0])*100))

print("Breast Cancer:")
cancer = ds.load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=.20)
clsfyr = mlp.train(20, X_train, y_train)
results = mlp.classify(clsfyr, X_test)
print("accuracy: {0:.2f}%\n".format(compare(results, y_test)*100))

print("Mushrooms:")
# UCI's mushroom dataset found at
# https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/
mushroom = parseCSV('agaricus-lepiota.data')
# We use only use 3000 of the mushroom data points so it won't take so long
# We also will convert the letters in the data to ascii so it can quantified
target = mushroom[:3000,0]
target2 = [0]*3000
for i,t in enumerate(target):
    if t == 'e':
        target2[i] = int(0);
    else:
        target2[i] = int(1);
data = [[int(ord(a)) for a in d[1:]] for d in mushroom[:3000]]
X_train, X_test, y_train, y_test = train_test_split(data, target2, test_size=.20)
clsfyr = mlp.train(20, X_train, y_train)
results = mlp.classify(clsfyr, X_test)
print("accuracy: {0:.2f}%\n".format(compare(results, y_test)*100))
