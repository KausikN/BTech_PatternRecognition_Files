'''
Assignment Q1
Q1. Calculate the distance between the two normalized histograms H1 and H2 using each of the
following methods:
I. KL Distance
II. Bhattacharyya Distance
H1 = [ 0.24, 0.2, 0.16, 0.12, 0.08, 0.04, 0.12, 0.04]
H2 = [ 0.22, 0.23, 0.16, 0.13, 0.11, 0.08, 0.05, 0.02]
'''

# Imports
import numpy as np

# Main Functions
def KLDistance(P1, P2):
    dist = 0

    P1 = np.array(P1)
    P2 = np.array(P2)

    dist = np.sum(P1*np.log(np.divide(P1, P2)))

    return dist

def BhattacharyyaDistance(P1, P2):
    dist = 0

    P1 = np.array(P1)
    P2 = np.array(P2)

    dist = -np.log(np.sum(np.sqrt(np.multiply((P1 / np.sum(P1)), (P2 / np.sum(P2))))))

    return dist

# Driver Code
# Params
H1 = [0.24, 0.2, 0.16, 0.12, 0.08, 0.04, 0.12, 0.04]
H2 = [0.22, 0.23, 0.16, 0.13, 0.11, 0.08, 0.05, 0.02]
# Params

# RunCode
print("")
print("Histogram 1:\n", H1)
print("Histogram 2:\n", H2)
print("")

# KL Distance
KLDist = KLDistance(H1, H2)
print("KL Distance:", KLDist)

print("")

# Bhattacharyya Distance
BhattacharyyaDist = BhattacharyyaDistance(H1, H2)
print("Bhattacharyya Distance:", BhattacharyyaDist)