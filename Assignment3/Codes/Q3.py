'''
Assignment Q3
Q3. Find and plot the decision boundary between class ω1 and ω2. Assume P(ω1) = P(ω2).
ω1 = [2,6; 3,4; 3,8; 4,6]
ω2 = [3,0; 1,-2; 3,-4; 5,-2]
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
    ConfMatrix_df.index = Classes
    ConfMatrix_df.columns = Classes

    return ConfMatrix_df

# Driver Code
# Params

# Params

# RunCode