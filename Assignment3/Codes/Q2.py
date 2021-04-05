'''
Assignment Q2
Q2. For the same dataset (2 classes, male and female)
a) Use LDA to reduce the dimension from d to d’. (Here d=128)
b) Choose the direction W to reduce the dimension d’ (select appropriate d’).
c) Use d’ features to classify the test cases (any classification algorithm will do, Bayes
classifier, minimum distance classifier, and so on).
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
    dataset.rename(columns={'Unnamed: 0':'SNo','Unnamed: 1':'Gender'},inplace=True)
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

def LDA(dataset_Matrix, reducedDim=None, adaptiveDimPartial=0.95, plot=False, display=False):
    Genders = ['male', 'female']
    for i in Genders:
        mean_vectors.append(np.mean(dataset_Matrix[Target == i], axis=0))
        if display:
            print("Mean Vector of class", i, ":", mean_vectors)

    dim = dataset_Matrix.shape[1]

    S_W = np.zeros((dim, dim))
    for cl, mv in zip(Genders, mean_vectors):
        class_sc_mat = np.zeros((dim, dim))                  # scatter matrix for every class
        for row in dataset_Matrix[Target == cl]:
            row, mv = row.reshape(dim, 1), mv.reshape(dim, 1) # make column vectors
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat

    if display:
        print("Within Class Scatter Matrix :\n", S_W)

    overall_mean = np.mean(dataset_Matrix, axis=0)

    S_B = np.zeros((dim, dim))
    for cl,mean_vec in zip(Genders, mean_vectors):  
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
    if minDims > len(Genders) - 1:
        minDims = len(Genders) - 1

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
datasetPath = 'Assignment3/Data/gender_feature_vectors.csv'

plot = False
display = True
# Params

# RunCode
# Load Dataset
dataset = LoadDataset(datasetPath)

# MAIN CODE FOR LDA BEGINS #
mean_vectors = []

Target = dataset['Gender'].values
dataset_Matrix = dataset.drop(labels=['SNo', 'Gender'], axis=1).to_numpy()

dataset_Matrix_LDA = LDA(dataset_Matrix, reducedDim=None, adaptiveDimPartial=0.95, plot=plot, display=display)

# Prepare Data
Classes = np.array(dataset['Gender'].to_numpy() == 'male', dtype=int) # 1 for male, 0 for female

testCountPerClass = 10

X_male = dataset_Matrix_LDA[Classes == 1]
X_female = dataset_Matrix_LDA[Classes == 0]

X_train = np.array(list(X_male[testCountPerClass:]) + list(X_female[testCountPerClass:]))
Y_train = np.array(list([1]*(X_male[testCountPerClass:].shape[0])) + list([0]*(X_female[testCountPerClass:].shape[0])))

X_test = np.array(list(X_male[:testCountPerClass]) + list(X_female[:testCountPerClass]))
Y_test = np.array(list([1]*testCountPerClass) + list([0]*testCountPerClass))

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