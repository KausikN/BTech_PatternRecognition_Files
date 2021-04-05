'''
Assignment Q5
Q5. Fisherfaces- Face classification using LDA (40 classes)
e) Use the following “face.csv” file to classify the faces of 40 different people.
f) Do not use the in-built function for implementing LDA.
g) Use appropriate classifier taught in class (any classification algorithm taught in class
like Bayes classifier, minimum distance classifier, and so on )
h) Refer to the following link for a description of the dataset:
https://towardsdatascience.com/eigenfaces-face-classification-in-python-7b8d2af3d3
'''

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, plot_implicit

import Utils

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

def LDA(dataset_Matrix, Target, reducedDim=None, adaptiveDimPartial=0.95, plot=False, display=False):
    Classes = list(np.unique(Target))
    for i in Classes:
        mean_vectors.append(np.mean(dataset_Matrix[Target == i], axis=0))
        if display:
            print("Mean Vector of class", i, ":", mean_vectors)

    dim = dataset_Matrix.shape[1]

    S_W = np.zeros((dim, dim))
    for cl, mv in zip(Classes, mean_vectors):
        class_sc_mat = np.zeros((dim, dim))                  # scatter matrix for every class
        for row in dataset_Matrix[Target == cl]:
            row, mv = row.reshape(dim, 1), mv.reshape(dim, 1) # make column vectors
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat

    if display:
        print("Within Class Scatter Matrix :\n", S_W)

    overall_mean = np.mean(dataset_Matrix, axis=0)

    S_B = np.zeros((dim, dim))
    for cl,mean_vec in zip(Classes, mean_vectors):  
        n = dataset_Matrix[Target == cl,:].shape[0]
        mean_vec = mean_vec.reshape(dim, 1) # make column vector
        overall_mean = overall_mean.reshape(dim, 1) # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

    if display:
        print('between-class Scatter Matrix:\n', S_B)

    eig_vals, eig_vecs = la.eig(np.linalg.inv(S_W).dot(S_B))
    eig_vals = eig_vals.real
    # for i in range(len(eig_vals)):
    #     eigvec_sc = eig_vecs[:,i].reshape(dim, 1)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # print(eig_pairs[0][1].real)
    if display:
        print('Eigenvalues in decreasing order:\n')
        for eP in eig_pairs:
            print(eP[0])

    print('Variance explained:\n')
    eigv_sum = sum(eig_vals)
    eigenVals = []
    for i, j in enumerate(eig_pairs):
        eigenVals.append(j[0])
        # print("Eigenvalue", i+1, ":", (j[0]*100/eigv_sum).real, "%")
    eigenVals = np.array(eigenVals)

    # Get Minimum Dimensions Count
    minDims = reducedDim
    if minDims is None:
        minDims = GetMinDimForSum(eigenVals, partial=adaptiveDimPartial)
    if minDims > len(Classes) - 1:
        minDims = len(Classes) - 1

    print("Selected Reduced Dimensions:", minDims)
    print()

    if plot:
        PlotVals(eigenVals[::], plots=[False, True, False])

    W = eig_pairs[0][1].reshape(dim, 1)
    for i in range(1, minDims):
        W = np.hstack((W, eig_pairs[i][1].reshape(dim, 1)))
        
    # print('Matrix W:\n', W.real)

    dataset_Matrix_LDA = dataset_Matrix.dot(W).real   #Converted Data_Matrix is dataset_Matrix_LDA
    if display:
        print("Dataset Matrix Shape", dataset_Matrix_LDA.shape, "\n after LDA\n", dataset_Matrix_LDA)

    return dataset_Matrix_LDA

# Driver Code
# Params
datasetPath = 'Assignment3/Data/face.csv'

plot = False
display = False
# Params

# RunCode
# Load Dataset
dataset = LoadDataset(datasetPath)

# MAIN CODE FOR LDA BEGINS #
mean_vectors = []

Target = dataset['target'].values
dataset_Matrix = dataset.drop(labels=['target'], axis=1).to_numpy()

dataset_Matrix_LDA = LDA(dataset_Matrix, Target, reducedDim=None, adaptiveDimPartial=0.95, plot=plot, display=display)

# import pickle
# pickle.dump(dataset_Matrix_LDA, open('Assignment3/Data/Q5LDAPoints.p', 'wb'))
# dataset_Matrix_LDA = pickle.load(open('Assignment3/Data/Q5LDAPoints.p', 'rb'))

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
testPoints = np.zeros((dataset_Matrix_LDA.shape[0]), dtype=bool)
testPoints[testIndices] = True

X_train = dataset_Matrix_LDA[~testPoints]
Y_train = dataset['target'][~testPoints]

X_test = dataset_Matrix_LDA[testPoints]
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