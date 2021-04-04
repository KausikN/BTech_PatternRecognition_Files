'''
Utils Functions
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, plot_implicit
import sympy

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Main Functionsd
def NaiveBayesClassifier(X_train, y_train):
    gnb = GaussianNB()
    classifier = gnb.fit(X_train, y_train)
    return classifier

def CovarianceMatrix(Pts):
    CM = np.cov(np.transpose(Pts))
    return CM



# Driver Code