'''
Assignment Q3
Q3. Find and plot the decision boundary between class ω1 and ω2. Assume P(ω1) = P(ω2).
ω1 = [2,6; 3,4; 3,8; 4,6]
ω2 = [3,0; 1,-2; 3,-4; 5,-2]
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, plot_implicit

import Utils

# Main Functions


# Driver Code
# Params
W1 = [[2, 6], [3, 4], [3, 8], [4, 6]]
P_W_1 = 0.5
W2 = [[3, 0], [1, -2], [3, -4], [5, -2]]
P_W_2 = 0.5

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