'''
Assignment Q2
Q2. Given (hq-ht)T = [0.5, 0.5, -0.5, -0.25, -0.25]
A = [
    [1, 0.135, 0.195, 0.137, 0.157],
    [0.135, 1, 0.2, 0.309, 0.143],
    [0.195, 0.2, 1, 0.157, 0.122],
    [0.137, 0.309, 0.157, 1, 0.195],
    [0.157, 0.143, 0.122, 0.195, 1]
]
Find the quadratic form distance.
'''

# Imports
import numpy as np

# Main Functions
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
diff = [0.5, 0.5, -0.5, -0.25, -0.25]
A = [
    [1, 0.135, 0.195, 0.137, 0.157],
    [0.135, 1, 0.2, 0.309, 0.143],
    [0.195, 0.2, 1, 0.157, 0.122],
    [0.137, 0.309, 0.157, 1, 0.195],
    [0.157, 0.143, 0.122, 0.195, 1]
]
# Params

# RunCode
print("")
print("Diff:\n", diff)
print("Transformation Matrix:\n", A)
print("")

# Quadratic Form Distance
QuadraticFormDist = QuadraticFormDistance(diff, [0]*len(diff), A)
print("Quadratic Form Distance:", QuadraticFormDist)