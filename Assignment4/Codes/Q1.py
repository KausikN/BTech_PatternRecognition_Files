'''
Assignment Q1
Q1. Train a single perceptron and SVM to learn an AND gate with two inputs x1 and x2. Assume that all
the weights of the perceptron are initialized as 0. Show the calulation for each step and also draw the decision
boundary for each updation.
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
    dataset.rename(columns = {'Unnamed: 0': 'SNo', 'Unnamed: 1': 'Gender'}, inplace = True)
    return dataset

def PlotVals(vals, plots=[True, True, True]):
    if plots[0]:
        plt.scatter(range(vals.shape[0]), vals)
    if plots[1]:
        plt.plot(range(vals.shape[0]), vals)
    if plots[2]:
        plt.bar(range(vals.shape[0]), vals, width=0.9)
    plt.show()

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
Accuracy = (Correct / len(Y_Pred))* 100

print("Correct:", Correct)
print("Wrong:", Wrong)

print("Accuracy : ", Accuracy)