'''
Assignment Q3
Q3. In the given I set of images from poly1.png to poly14.png, let poly1 to poly 7 belong to class 1 and poly 8 to
poly 14 belong to class 2. Assume that all the weights of the perceptron are initialized as 0 with the learning
rate of 0.01.
• Identify two discriminant features x1 and x2 for the two target classes ω={ω1, ω2}. Here, ω1 - class 1 and ω2 - class 2.
• Generate an input feature vector X for all the images mapping them to a corresponding taget classes ωi, where i E (1, 2).
• Train a single perceptron and SVM to learn the feature vector X mapping to ω.
• Plot and draw the final decision boundary separating the three classes
'''

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import symbols, plot_implicit

import Utils

# Main Functions
def ConfusionMatrix(Y_test, Y_Pred, Classes):
    dim = len(Classes)
    Conf_Matrix = np.zeros((dim, dim))
    # row- True ; column- predicted
    for Test, pred in zip(Y_test, Y_Pred):
        Conf_Matrix[Test][pred] += 1
    Conf_Matrix /= Y_test.shape[0]

    columns = []
    indices = []
    for c in Classes:
        columns.append("Predicted " + str(c))
        indices.append("Actual " + str(c))

    ConfMatrix_df = pd.DataFrame(Conf_Matrix, columns=columns, index=indices)

    return ConfMatrix_df

# Driver Code
# Params

# Params

# RunCode