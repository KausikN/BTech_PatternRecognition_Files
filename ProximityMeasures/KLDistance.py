"""
KL Distance
"""

# Imports
import numpy as np

# Driver Code
def KLDistance(P1, P2):
    dist = 0

    P1 = np.array(P1) / np.sum(P1)
    P2 = np.array(P2) / np.sum(P2)

    dist = np.sum(P1*np.log2(np.divide(P1, P2)))

    return dist

# Driver Code
# Params
P2 = [1/3, 1/2, 1/2]
P1 = [1/2, 1/3, 1/2]
# Params

# RunCode
dist = KLDistance(P1, P2)
print("P1:\n", P1)
print("P2:\n", P2)
print("KL Distance:\n", dist)