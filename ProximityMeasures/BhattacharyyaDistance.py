"""
Bhattacharyya Distance
"""

# Imports
import numpy as np

# Driver Code
def BhattacharyyaDistance(P1, P2):
    dist = 0

    P1 = np.array(P1)
    P2 = np.array(P2)

    dist = 1 - np.sum(np.sqrt(np.multiply((P1 / np.sum(P1)), (P2 / np.sum(P2)))))

    return dist

# Driver Code
# Params
P1 = [0.2, 0.5]
P2 = [0.2, 0.4]
# Params

# RunCode
dist = BhattacharyyaDistance(P1, P2)
print("P1:\n", P1)
print("P2:\n", P2)
print("Bhattacharyya Distance:\n", dist)