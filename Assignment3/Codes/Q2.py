'''
Assignment Q2
Q2. Find and plot the decision boundary between class ω1 and ω2. Assume P(ω1) =0.3; P(ω2)=0.7
ω1 = [1,-1; 2,-5; 3,-6; 4,-10; 5,-12; 6,-15]
ω2 = [-1,1; -2,5; -3,6; -4,10, -5,12; -6, 15]
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, plot_implicit

import Utils

# Main Functions


# Driver Code
# Params
W1 = [[1, -1], [2, -5], [3, -6], [4, -10], [5, -12], [6, -15]]
P_W_1 = 0.3
W2 = [[-1, 1], [-2, 5], [-3, 6], [-4, 10], [-5, 12], [-6, 15]]
P_W_2 = 0.7

linePadding = 2
lineResolution = 100
# Params

# RunCode
boundary = Utils.GetBoundary(W1+W2)
rangeValues = [boundary[0][0]-linePadding, boundary[0][1]+linePadding, lineResolution]

eq, case = Utils.DecisionBoundaryEquation_2Class(W1, W2, P_W_1, P_W_2)
print("Equation of decision boundary:")
print(eq, "= 0")
# Utils.PlotEquation(eq)
CurvesPts = Utils.Equation2Points(eq, rangeValues=rangeValues)
Utils.PlotPoints([W1, W2], colors=['red', 'blue'])
Utils.PlotLines(CurvesPts, colors=['black']*len(CurvesPts))
plt.title('Decision Boundary')
plt.show()