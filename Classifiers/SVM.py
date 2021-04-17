'''
SVM Code
'''

# Imports
import numpy as np
from cvxopt import matrix, solvers


# Main Functions
def poly_kernel(x, z, degree, intercept):
        return np.power(np.matmul(x, z.T) + intercept, degree)

def gaussian_kernel(x, z, sigma):
    n = x.shape[0]
    m = z.shape[0]
    xx = np.dot(np.sum(np.power(x, 2), 1).reshape(n, 1), np.ones((1, m)))
    zz = np.dot(np.sum(np.power(z, 2), 1).reshape(m, 1), np.ones((1, n)))     
    return np.exp(-(xx + zz.T - 2 * np.dot(x, z.T)) / (2 * sigma ** 2))

def linear_kernel(x, z):
    return np.matmul(x, z.T)

# Driver Code
X = [
    [3, 1],
    [3, -1],
    [6, 1],
    [6, -1],
    [1, 0],
    [0, 1],
    [0, -1],
    [-1, 0]
]

Y = [1, 1, 1, 1, -1, -1, -1, -1]

X = np.array(X)
Y = np.array(Y).reshape(8, 1)

m = len(X)

print(np.dot(Y,Y.T))
print(np.dot(X,X.T))
P = matrix(np.dot(Y, Y.T) * np.dot(X, X.T))
print(P)
q = matrix(np.ones(m) * -1)
g1 = np.asarray(np.diag(np.ones(m) * -1))
# g2 = np.asarray(np.diag(np.ones(m)))
# G = matrix(np.append(g1, g2, axis=0))
print(np.array(g1).shape)
h = matrix(np.zeros(m))
A = np.reshape((Y.T), (1,m))
b=[[0]]
# b = np.array(b).reshape(m,1)

P = matrix(P,(m,m),'d')
A = matrix(A,(1,m),'d')
g1 = matrix(g1,(m,m),'d')
b = matrix(b,(1,1),'d')

sol = solvers.qp(P, q, g1, h, A, b)
alpha = np.array(sol['x'])
print(alpha)

ind = (alpha > 1e-4).flatten()
print(ind)

W = np.dot(np.transpose(alpha*Y),X)
print("W:\n", W)

X_0 = np.array(X[0]).reshape(1, 2)

W0 = []
for i in range(m):
    if ind[i] == True:
        W0 = 1 - np.dot(X[i], W.T)
        break
print(W0)

Z = [1, -2]

Z = np.array(Z).reshape(1, 2)

print(Z.shape, W.T.shape)
Pred = (np.dot(Z, W.T) + W0[0]) > 0
print("Z:", Z)
print("Pred:", Pred)