'''
Assignment Q1
Q1. Consider the 128- dimensional feature vectors (d=128) given in the “gender_feature
_vectors.csv” file. (2 classes, male and female)

a) Use PCA to reduce the dimension from d to d’. (Here d=128)
b) Display the eigenvalue based on increasing order, select the d’ of the corresponding
eigenvector which is the appropriate dimension d’ ( select d’ S.T first 95% of λ values
of the covariance matrix are considered).
c) Use d’ features to classify the test cases (any classification algorithm taught in class
like Bayes classifier, minimum distance classifier, and so on)

Dataset Specifications:
Total number of samples = 800
Number of classes = 2 (labeled as “male” and “female”)
Samples from “1 to 400” belongs to class “male”
Samples from “401 to 800” belongs to class “female”
Number of samples per class = 400
Number of dimensions = 128
Use the following information to design classifier:
Number of test cases ( first 10 in each class) = 20
Number of training feature vectors ( remaining 390 in each class) = 390
Number of reduced dimensions = d’ (map 128 to d’ features vector)
'''

# Imports
import numpy as np
import pandas as pd
import scipy.linalg as la
import matplotlib.pyplot as plt
from sympy import symbols, plot_implicit

import Utils

# Main Functions
def LoadDataset(path='gender_featurevectors.csv'):
    dataset = pd.read_csv(path)
    return dataset

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
datasetPath = 'Assignment3/Data/gender_feature_vectors.csv'

plot = False
display = True
# Params

# RunCode
# Load Dataset
dataset = LoadDataset(datasetPath)
if display:
    print("Dataset:\n", dataset.head())
    print()

dataset_Matrix = dataset.drop(labels=['SNo', 'Gender'], axis=1).to_numpy()

dataset_PCATransformed = PCA(dataset_Matrix, reducedDim=None, adaptiveDimPartial=0.95, plot=plot, display=display)

# Classify using Naive Bayes
# Params
testCountPerClass = 10
# Params

# Prepare Data
Classes = np.array(dataset['Gender'].to_numpy() == 'male', dtype=int) # 1 for male, 0 for female
if display:
    print("Classes:", Classes.shape, "\n", Classes)
    print()

X_male = dataset_PCATransformed[Classes == 1]
X_female = dataset_PCATransformed[Classes == 0]

X_train = np.array(list(X_male[testCountPerClass:]) + list(X_female[testCountPerClass:]))
Y_train = np.array(list([1]*(X_male[testCountPerClass:].shape[0])) + list([0]*(X_female[testCountPerClass:].shape[0])))

X_test = np.array(list(X_male[:testCountPerClass]) + list(X_female[:testCountPerClass]))
Y_test = np.array(list([1]*testCountPerClass) + list([0]*testCountPerClass))

# Classify and Predict
Classifier = Utils.NaiveBayesClassifier(X_train, Y_train)
Y_Pred = Classifier.predict(X_test)
if display:
    print("Predicted Values for test:", Y_Pred.shape, "\n", Y_Pred)
    print()

Correct = np.count_nonzero(Y_test == Y_Pred)
Wrong = Y_test.shape[0] - Correct

print("Correct:", Correct)
print("Wrong:", Wrong)