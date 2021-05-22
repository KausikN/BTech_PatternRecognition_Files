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
P1 = [0.195, 0.122, 0.286, 0.017, 0.282, 0.016, 0.082]
P2 = [0.232, 0.181, 0.075, 0.063, 0.196, 0.192, 0.061]
# Params

# RunCode
dist = KLDistance(P1, P2)
print("P1:\n", P1)
print("P2:\n", P2)
print("KL Distance:\n", dist)