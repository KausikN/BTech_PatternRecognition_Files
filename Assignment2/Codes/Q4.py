'''
Assignment Q4
Q4. Implement Bayes Classifier for Iris Dataset. 
Dataset Specifications: 
Total number of samples = 150
Number of classes = 3 (Iris setosa, Iris virginica, and Iris versicolor)
Number of samples in each class = 50 
Use the following information to design classifier: 
    Number of training feature vectors ( first 40 in each class) = 40
    Number of test feature vectors ( remaining 10 in each class) = 10
    Number of dimensions = 4
    Feature vector = <sepal length, sepal width, petal length, petal width>
If the samples follow a multivariate normal density, find the accuracy of classification for the test 
feature vectors.
'''

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, plot_implicit
import sympy
from Utils import *

# Main Functions
def P_X_wi(det_sq, X,mean, cov_inv, d=4):
    coeff = 1/(((2*np.pi)**d/2)*det_sq)
    prob = coeff*np.exp(-0.5*np.matmul(np.matmul(X-mean,cov_inv),np.transpose(X-mean)))
    return prob

# Driver Code
# Params
path = 'Assignment2/Data/Iris_dataset.csv'
# Params

# RnCode
data = pd.read_csv(path)

# P(w1 |X),P(w2 |X),P(w3 |X) --> Find which is max for the given X
# P(wi |X) = P(X |wi)*P(wi)   {Divided by P(X) can be ignored}
# P(wi) = n/N
# P(X |wi) = multivariate pdf formula

Test_variety = data.iloc[40:50,4]
Test_variety = Test_variety.append(data.iloc[90:100,4])
Test_variety = Test_variety.append(data.iloc[140:150,4])
Test = data.iloc[40:50,0:4]
Test = Test.append(data.iloc[90:100,0:4])
Test = Test.append(data.iloc[140:150,0:4])

data_X = data.iloc[0:40]
data_X = data_X.append(data.iloc[50:90])
data_X = data_X.append(data.iloc[100:140])

## The 3 flowers are "Setosa", "Versi", "Virginica"
Setosa_data = data_X.iloc[0:40]
Versi_Color_data = data_X.iloc[40:80]
Virginica_data = data_X.iloc[80:120]

Setosa_data_1 = data.iloc[0:40]
Versi_Color_data_1 = data.iloc[50:90]
Virginica_data_1 = data.iloc[100:140]

# Probabilities
setosa_mean = np.mean(Setosa_data)
versi_mean = np.mean(Versi_Color_data)
virginica_mean = np.mean(Virginica_data)

setosa_Z = Setosa_data - setosa_mean
versi_Z = Versi_Color_data - versi_mean
virginica_Z = Virginica_data - virginica_mean

setosa_cov = np.cov(np.transpose(setosa_Z))
versi_cov = np.cov(np.transpose(versi_Z))
virginica_cov = np.cov(np.transpose(virginica_Z))

setosa_A = np.linalg.inv(setosa_cov)
versi_A = np.linalg.inv(versi_cov)
virginica_A = np.linalg.inv(virginica_cov)

setosa_det_sq = (np.linalg.det(setosa_cov))**0.5
versi_det_sq = (np.linalg.det(versi_cov))**0.5
virginica_det_sq = (np.linalg.det(virginica_cov))**0.5

P_X_w1 = []
P_X_w2 = []
P_X_w3 = []
# print(setosa_mean.shape)
for i in range(len(Test)):
    P_X_w1.append(P_X_wi(setosa_det_sq,Test.iloc[i],setosa_mean,setosa_A))
    P_X_w2.append(P_X_wi(versi_det_sq,Test.iloc[i],versi_mean,versi_A))
    P_X_w3.append(P_X_wi(virginica_det_sq,Test.iloc[i],virginica_mean,virginica_A))

count=0
count_w1, count_w2, count_w3 = 0, 0, 0
Category = ["Setosa", "Versicolor","Virginica"]
# print(len(Test_variety))
for i in range(len(P_X_w1)):
    if Test_variety.iloc[i]==Category[np.argmax([P_X_w1[i], P_X_w2[i], P_X_w3[i]])]:
        count+=1
    if Test_variety.iloc[i]==Category[np.argmax([P_X_w1[i]])]:
        count_w1+=1
    if Test_variety.iloc[i]==Category[np.argmax([P_X_w2[i]])]:
        count_w2+=1
    if Test_variety.iloc[i]==Category[np.argmax([P_X_w3[i]])]:
        count_w3+=1

N = len(data_X)
num_variety = data_X.groupby(by='variety').agg('count')
n_1 = num_variety['sepal.length'][0]   # In this case n_1=n_2=n_3
P_w1 = P_w2 = P_w3 = n_1/N
data_X = data_X.drop(columns='variety')

print("The overall performance accuracy is {x}%".format(x=100*count/len(Test_variety)))
print("The accuracy for class w1(Setosa) is {x}%".format(x=100*count_w1/10))
print("The accuracy for class w2(Versicolor) is {x}%".format(x=100*count_w2/10))
print("The accuracy for class w3(Virginica) is {x}%".format(x=100*count_w3/10), end="\n\n")

# Get Pts
featurenames = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
Setosa_pts = np.dstack((Setosa_data_1['sepal.length'], Setosa_data_1['sepal.width'], Setosa_data_1['petal.length'], Setosa_data_1['petal.width']))[0]
Versi_Color_pts = np.dstack((Versi_Color_data_1['sepal.length'], Versi_Color_data_1['sepal.width'], Versi_Color_data_1['petal.length'], Versi_Color_data_1['petal.width']))[0]
Virginica_pts = np.dstack((Virginica_data_1['sepal.length'], Virginica_data_1['sepal.width'], Virginica_data_1['petal.length'], Virginica_data_1['petal.width']))[0]

# Discriminant Functions
Setosa_eq = DiscriminantFunctionEquation(Setosa_pts, 1/3)
Versi_Color_eq = DiscriminantFunctionEquation(Versi_Color_pts, 1/3)
Virginica_eq = DiscriminantFunctionEquation(Virginica_pts, 1/3)
Setosa_eqpoly = sympy.poly(Setosa_eq)
Versi_Color_eqpoly = sympy.poly(Versi_Color_eq)
Virginica_eqpoly = sympy.poly(Virginica_eq)
print("Sentosa Discriminant Func:\n", Setosa_eqpoly)
print()
print("Versi_Color Discriminant Func:\n", Versi_Color_eqpoly)
print()
print("Virginica Discriminant Func:\n", Virginica_eqpoly)

print()

# Decision Boundary
DB_Setosa_VersiColor = Setosa_eq - Versi_Color_eq
DB_VersiColor_Virginica = Versi_Color_eq - Virginica_eq
DB_Virginica_Setosa = Setosa_eq - Virginica_eq

DB_Setosa_VersiColorpoly = sympy.poly(DB_Setosa_VersiColor)
DB_VersiColor_Virginicapoly = sympy.poly(DB_VersiColor_Virginica)
DB_Virginica_Setosapoly = sympy.poly(DB_Virginica_Setosa)
print("Decision Boundary Sentosa vs VersiColor:\n", DB_Setosa_VersiColorpoly, "= 0")
print()
print("Decision Boundary VersiColor vs Virginica:\n", DB_VersiColor_Virginicapoly, "= 0")
print()
print("Decision Boundary Sentosa vs Virginica:\n", DB_Virginica_Setosapoly, "= 0")

print()