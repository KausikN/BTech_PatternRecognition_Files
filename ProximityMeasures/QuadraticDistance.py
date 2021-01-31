"""
Quadratic Form Distance
"""

# Imports
import numpy as np

# Driver Code
def QuadraticFormDistance(P1, P2, TransformationMatrix):
    dist = 0

    A = np.array(TransformationMatrix)
    P1 = np.transpose([P1])
    P2 = np.transpose([P2])
    diff = P1 - P2

    dist = np.matmul(np.matmul(np.transpose(diff), A), diff)[0, 0]

    return dist

# Driver Code
# Params
P1 = [0, 0, 0]
P2 = [2, -4, 3]
A = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]
# Params

# RunCode
dist = QuadraticFormDistance(P1, P2, A)
print("P1:\n", P1)
print("P2:\n", P2)
print("Transformation Matrix:\n", A)
print("Quadratic Form Distance:\n", dist)