"""
Minkowski Metric LP Norm Distance
"""

# Imports
import numpy as np

# Driver Code
def LPNormDistance(P1, P2, PNorm):
    dist = 0
    for x, y in zip(P1, P2):
        dist += (abs(x - y))**(PNorm)
    dist = dist ** (1/PNorm)
    return dist

def ManhattanDistance(P1, P2):
    dist = LPNormDistance(P1, P2, 1)
    return dist

def L_Inf_Distance(P1, P2):
    dist = np.max(np.abs(np.array(P1) - np.array(P2)))
    return dist

def L_MinusInf_Distance(P1, P2):
    dist = np.min(np.abs(np.array(P1) - np.array(P2)))
    return dist

# Driver Code
# Params
P1 = [0, 0, 0]
P2 = [2, -4, 3]
PNorm = 2
# Params

# RunCode
dist = LPNormDistance(P1, P2, PNorm)
print("LPNorm Distance for P:", PNorm, "is", dist)