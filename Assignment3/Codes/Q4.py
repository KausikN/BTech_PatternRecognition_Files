'''
Assignment Q4
Q4. Eigenfaces-Face classification using PCA (40 classes)
a) Use the following “face.csv” file to classify the faces of 40 different people.
b) Do not use the in-built function for implementing PCA.
c) Use appropriate classifier taught in class (any classification algorithm taught in class
like Bayes classifier, minimum distance classifier, and so on )
d) Refer to the following link for a description of the dataset:
https://towardsdatascience.com/eigenfaces-face-classification-in-python-7b8d2af3d3e
'''

# Imports
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from sympy import symbols, plot_implicit

import Utils

# Main Functions
def LoadDataset(path='face.csv'):
    dataset = pd.read_csv(path)
    return dataset

def GetEigenValsVecs(mat):
    eigenVals, eigenVecs = la.eig(mat)
    eVals = eigenVals.real
    eVecs = np.array(eigenVecs.real).T
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
datasetPath = 'Assignment3/Data/face.csv'

plot = False
display = True
# Params

# RunCode
# Load Dataset
dataset = LoadDataset(datasetPath)
if display:
    print("Dataset:\n", dataset.head())
    print()

dataset_Matrix = dataset.drop(labels=['target'], axis=1).to_numpy()

dataset_PCATransformed = PCA(dataset_Matrix, reducedDim=None, adaptiveDimPartial=0.95, plot=plot, display=display)

# import pickle
# pickle.dump(dataset_PCATransformed, open('Assignment3/Data/Q4PCAPoints.p', 'wb'))
# dataset_PCATransformed = pickle.load(open('Assignment3/Data/Q4PCAPoints.p', 'rb'))

# Classify using Naive Bayes
# Params
testCount = 40
# Params

# Prepare Data
Classes = np.array(dataset['target'].to_numpy(), dtype=int)
if display:
    print("Classes:", Classes.shape, "\n", Classes)
    print()

testIndices = np.random.choice(list(range(dataset.shape[0])), replace=False, size=testCount)
testPoints = np.zeros((dataset_PCATransformed.shape[0]), dtype=bool)
testPoints[testIndices] = True

X_train = dataset_PCATransformed[~testPoints]
Y_train = dataset['target'][~testPoints]

X_test = dataset_PCATransformed[testPoints]
Y_test = dataset['target'][testPoints]

# Classify and Predict
Classifier = Utils.NaiveBayesClassifier(X_train, Y_train)
Y_Pred = Classifier.predict(X_test)
if display:
    print("Predicted Values for test:", Y_Pred.shape, "\n", Y_Pred)
    print()

Correct = np.count_nonzero(Y_test == Y_Pred)
Wrong = Y_test.shape[0] - Correct
Accuracy = (Correct / len(Y_Pred))* 100

print("Correct:", Correct)
print("Wrong:", Wrong)

print("Accuracy : ", Accuracy)