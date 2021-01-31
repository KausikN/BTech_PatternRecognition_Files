"""
Cosine Distance
"""

# Imports
import numpy as np

# Driver Code
def CosineDistance(P1, P2):
    dist = 0

    # Convert both to same length
    if not len(P1) == len(P2):
        if len(P1) > len(P2):
            P2 = P2 + [0]*(len(P1)-len(P2))
        else:
            P1 = P1 + [0]*(len(P2)-len(P1))

    u = np.array(P1)
    v = np.array(P2)

    dist = np.dot(u, v) / ((np.dot(u, u)**(0.5)) * (np.dot(v, v)**(0.5)))

    return dist

# Driver Code
# Params
P1 = [0.2, 0.5]
P2 = [0.2, 0.4]
# Params

# RunCode
dist = CosineDistance(P1, P2)
print("P1:\n", P1)
print("P2:\n", P2)
print("Cosine Distance:\n", dist)