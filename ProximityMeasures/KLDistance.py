"""
KL Distance
"""

# Imports
import numpy as np

# Driver Code
def KLDistance(P1, P2):
    dist = 0

    P1 = np.array(P1)
    P2 = np.array(P2)

    dist = np.sum(P1*np.log(np.divide(P1, P2)))

    return dist

# Driver Code
# Params
P1 = [0.2, 0.5]
P2 = [0.2, 0.4]
# Params

# RunCode
dist = KLDistance(P1, P2)
print("P1:\n", P1)
print("P2:\n", P2)
print("KL Distance:\n", dist)