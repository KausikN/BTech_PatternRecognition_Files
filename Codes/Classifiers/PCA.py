'''
PCA Dimensionality Reduction
'''

# Imports
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from sympy import symbols, plot_implicit

import Utils

# Main Functions
def GetEigenValsVecs(mat):
    eigenVals, eigenVecs = la.eig(mat)
    eVals = eigenVals.real
    eVecs = np.array(eigenVecs).T
    return eVals, eVecs

def SortEigens(vals, vecs):
    eVals, eVecs = zip(*sorted(zip(list(vals), list(vecs)), key=lambda x:x[0], reverse=True))
    eVals = np.array(eVals)
    eVecs = np.array(eVecs)
    return eVals, eVecs

def PlotVals(vals, plots=[True, True, True]):
    if plots[0]:
        plt.scatter(range(vals.shape[0]), vals)
    if plots[1]:
        plt.plot(range(vals.shape[0]), vals)
    if plots[2]:
        plt.bar(range(vals.shape[0]), vals, width=0.9)
    plt.show()

def GetMinDimForSum(vals, partial=0.95):
    valsSum = np.sum(vals)
    partialSum = valsSum * partial
    minDim = 0
    curSum = 0.0
    for i in range(vals.shape[0]):
        curSum += vals[i]
        minDim += 1
        if curSum >= partialSum:
            break
    return minDim

def PCA(dataset_Matrix, reducedDim=None, adaptiveDimPartial=0.95, plot=False, display=False):
    if display:
        print("Data points:", dataset_Matrix.shape)
        print()

    # Get Cov Matrix and Eigen Values and Vectors
    dataset_CM = Utils.CovarianceMatrix(dataset_Matrix)
    if display:
        print("Covariance Matrix:", dataset_CM.shape, "\n", dataset_CM)
        print()

    eigenVals, eigenVecs = GetEigenValsVecs(dataset_CM)
    eigenVals, eigenVecs = SortEigens(eigenVals, eigenVecs)
    if display:
        print("Eigen Values:", eigenVals.shape, "\n", eigenVals)
        print()
        print("Eigen Vectors:", eigenVecs.shape, "\n", eigenVecs)
        print()

    if plot:
        PlotVals(eigenVals[::], plots=[False, True, False])

    # Calculate Minimum dimensions to choose for 95%
    minDims = reducedDim
    if minDims is None:
        minDims = GetMinDimForSum(eigenVals, partial=adaptiveDimPartial)
    if display:
        print("Selected Reduced Dimensions:", minDims)
        print()

    # Form PCA Transform Matrix
    transformMatrix = eigenVecs[:minDims].T
    if display:
        print("Transform Matrix:", transformMatrix.shape, "\n", transformMatrix)
        print()

    # Reorient Dataset Points to new dimension
    dataset_PCATransformed = np.matmul(dataset_Matrix, transformMatrix)
    if display:
        print("Reduced Data values:", dataset_PCATransformed.shape, "\n", dataset_PCATransformed)
        print()

    return dataset_PCATransformed

# Driver Code
# Params
X = [
    [3, 1],
    [3, -1],
    [6, 1],
    [6, -1],
    [1, 0],
    [0, 1],
    [0, -1],
    [-1, 0]
]

Y = [1, 1, 1, 1, -1, -1, -1, -1]

plot = False
display = True
# Params

# RunCode
Target = np.array(Y)
dataset_Matrix = np.array(X)

if display:
    print("Dataset:\n", dataset_Matrix)
    print()

dataset_PCATransformed = PCA(dataset_Matrix, reducedDim=1, adaptiveDimPartial=0.95, plot=plot, display=display)

# Prepare Data
Classes = Target

X_1 = dataset_PCATransformed[Classes == 1]
X_0 = dataset_PCATransformed[Classes == -1]

X_train = np.array(list(X_1) + list(X_0))
Y_train = np.array(list([1]*(X_1.shape[0])) + list([-1]*(X_0.shape[0])))

X_test = [
    [2.1]
]
Y_test = [-1]

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Classify and Predict
Classifier = Utils.NaiveBayesClassifier(X_train, Y_train)
Y_Pred = Classifier.predict(X_test)
print("Predicted Values for test:", Y_Pred.shape, "\n", Y_Pred)
print()

Correct = np.count_nonzero(Y_test == Y_Pred)
Wrong = Y_test.shape[0] - Correct
Accuracy = (Correct / len(Y_Pred))* 100

print("Correct:", Correct)
print("Wrong:", Wrong)

print("Accuracy : ", Accuracy, "%")