'''
Assignment Q5
Q5. Use only two features: Petal Length and Petal Width, for 3 class classification and draw the 
decision boundary between them (2 dimension, 3 regions also called as multi-class problem)
'''

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, plot_implicit

import Utils

# Main Functions
def LoadIrisDataset_PetalData(path='Assignment2/Data/Iris_dataset.csv'):
    data = pd.read_csv(path)
    data_X = data.drop(columns=["sepal.length", "sepal.width"])
    return data_X

def IrisData2ClassPts(data):
    Classes = list(data['variety'].unique())
    Pts = {}
    for c in Classes:
        Pts[c] = []
    for i in range(data.shape[0]):
        Pts[data['variety'][i]].append([data['petal.length'][i], data['petal.width'][i]])
    return Pts, Classes

def Plot2Classes(C1, C2, W1, W2, P_W_1, P_W_2, linePadding, lineResolution, Colors=['red', 'blue', 'black'], display=True):
    if not type(linePadding) == list:
        linePadding = [linePadding, linePadding]
    boundary = Utils.GetBoundary(W1+W2)
    rangeValues = [boundary[0][0]+linePadding[0], boundary[0][1]+linePadding[1], lineResolution]

    eq, case = Utils.DecisionBoundaryEquation_2Class(W1, W2, P_W_1, P_W_2)
    print("Equation of decision boundary between", C1, "and", C2, ":")
    print(eq, "= 0")
    # Utils.PlotEquation(eq)
    CurvesPts = Utils.Equation2Points(eq, rangeValues=rangeValues)
    if display:
        Utils.PlotPoints([W1, W2], colors=[Colors[0], Colors[1]])
        Utils.PlotLines(CurvesPts + [[CurvesPts[0][0], CurvesPts[1][0]], [CurvesPts[0][-1], CurvesPts[1][-1]]], colors=[Colors[2]]*len(CurvesPts))

        title = 'Decision Boundary: ' + str(C1) + " vs ", str(C2)
        plt.title(title)
        plt.show()

    return CurvesPts

# Driver Code
# Params
path = 'Assignment2/Data/Iris_dataset.csv'

linePadding = [-2, 0]
lineResolution = 100000
# Params

# RunCode
# Load Iris Petal Data and get pts
data = LoadIrisDataset_PetalData(path)
Ws, Classes = IrisData2ClassPts(data)
print("Classes:", Classes)

CurvesPts = []
Colors = ['red', 'green', 'blue']
DBColors = ['yellow', 'purple', 'cyan']
k = 0
for i in range(len(Classes)):
    for j in range(i+1, len(Classes)):
        cps = Plot2Classes(Classes[i], Classes[j], Ws[Classes[i]], Ws[Classes[j]], 1/len(Classes), 1/len(Classes), linePadding, lineResolution, Colors=[Colors[i], Colors[j], DBColors[k]])
        CurvesPts.append(cps)
        k += 1

for i in range(len(Classes)):
    Utils.PlotPoints([Ws[Classes[i]]], colors=Colors[i])
for k in range(len(CurvesPts)):
    Utils.PlotLines(CurvesPts[k] + [[CurvesPts[k][0][0], CurvesPts[k][1][0]], [CurvesPts[k][0][-1], CurvesPts[k][1][-1]]], colors=[DBColors[k]]*len(CurvesPts[k]))
plt.title('Decision Boundary Combined')
plt.show()