'''
Single Layer Perceptron
Multi Layer Perceptron
'''

# Imports
import numpy as np

# Main Functions
def SLPerceptron(x):
    # LR is assumed to be 1
    W = np.array([0, 0, 0])
    k = 0
    n = len(x)
    chInd = 0
    start = True
    while start:
        if np.dot(W, np.transpose(x[k]))<=0:
            W = W + x[k]
            # print(W)
            chInd = k
        k = (k+1)%n
        if chInd==k:
            start=False
    return W

# Driver Code
X = np.array([
    [2, 2],
    [-1, -3],
    [-1, 2],
    [0, -1],
    [1, 3],
    [-1, -2],
    [1, -2],
    [-1, -1]
])
Y = np.array([
    [-1],
    [1],
    [-1],
    [1],
    [-1],
    [1],
    [1],
    [-1]
])

X_temp = X*Y
inp = np.hstack((X_temp, Y))

W_pred = SLPerceptron(inp)
print(W_pred)
thresh = W_pred[2]
# print(thresh)
W_pred = W_pred[:2].reshape((1,2))

# print(X)
Z = np.sum(W_pred*X, axis=1) + thresh
print("Equation of line is = {w1}*x1 + {w2}*x2 + {w3}".format(w1=W_pred[0][0],w2=W_pred[0][1],w3=thresh))