"""
Mahalanobis Distance
"""

# Imports
import numpy as np

# Driver Code
def MahalanobisDistance(Pts, targetPoint):
    Z = np.array(Pts)
    hq = np.array(targetPoint)

    Z_T = np.transpose(Z)

    CovMatrix = (1/(Z.shape[0]-1))*(np.matmul(Z_T, Z))

    print("Covariance Matrix")
    print(CovMatrix)

    invCM = np.linalg.inv(CovMatrix)

    print("Inverse Covariance Matrix")
    print(invCM)

    ht = targetPoint
    hq = np.mean(Z, axis=0)

    hd = hq - ht
    hd_T = np.transpose(hd)

    dist = np.matmul(np.matmul(hd_T, invCM), hd)

    return dist

# Driver Code
# Params
Pts = [
    [-4, -20, -11],
    [-2, -30, -7],
    [0, -10, -3],
    [1, 60, 6],
    [5, 0, 15]
]
targetPoint = np.array([68, 600, 40])
# Params

# RunCode
dist = MahalanobisDistance(Pts, targetPoint)
print("Mahalanobis Distance for target point:", targetPoint, "is", dist)