'''
Assignment Q4
Q4. Classify flower 0, 50, and 100 from the Iris Dataset (.csv file) attached along with the
assignment document into one of the three classes as given in dataset specification:
Dataset Specifications:
Total number of samples = 150
Number of classes = 3 (Iris setosa, Iris virginica, and Iris versicolor)
The number of samples in each class = 50.
Directions to classify:
1. Use features PetalLengthCm and PetalWidthCm only for classification.
2. Consider flowers 0,50 and 100 as test cases.
3. Plot the distribution of rest 147 sample points along with their classes( differentiate
classes with different colour). Consider PetalWidthCm along Y-axis and PetalLengthCm
along X-axis.
4. Capture the properties of the distribution and use suitable distance metrics to classify the
flowers 0,50 and 100 into one of the classes.
5. Print their class and plot the points on the previous plot with a marker differentiating the
three points.
'''

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

data = pd.read_csv("/content/drive/MyDrive/PR/Assignment_1/Data/Iris_dataset.csv")
# target = pd.DataFrame(data['variety'])
data_X = data.drop(columns=["sepal.length","sepal.width"])
Test = data_X.iloc[[0,50,100],[0,1]]
data_X = data_X.drop([0,50,100])
# target['variety'] = target['variety'].astype('category')
# target['variety'] = target['variety'].cat.codes

plt.scatter(data_X['petal.length'].iloc[0:49],data_X['petal.width'].iloc[0:49], color='r')
plt.scatter(data_X['petal.length'].iloc[49:98],data_X['petal.width'].iloc[49:98], color='#90ee90')
plt.scatter(data_X['petal.length'].iloc[98:147],data_X['petal.width'].iloc[98:147], color='yellow')
plt.show()

## The 3 flowers are "Setosa", "Versi", "Virginica"
Setosa_data = data_X.iloc[0:49,[0,1]]
Versi_Color_data = data_X.iloc[49:98,[0,1]]
Virginica_data = data_X.iloc[98:147,[0,1]]


setosa_mean = np.mean(Setosa_data)
versi_mean = np.mean(Versi_Color_data)
virginica_mean = np.mean(Virginica_data)

setosa_Z = Setosa_data - setosa_mean
versi_Z = Versi_Color_data - versi_mean
virginica_Z = Virginica_data - virginica_mean
N = len(setosa_Z)
setosa_cov = (1/(N-1))*np.dot(np.transpose(setosa_Z),setosa_Z)
versi_cov = (1/(N-1))*np.dot(np.transpose(versi_Z),versi_Z)
virginica_cov = (1/(N-1))*np.dot(np.transpose(virginica_Z),virginica_Z)

setosa_A = np.linalg.inv(setosa_cov)
versi_A = np.linalg.inv(versi_cov)
virginica_A = np.linalg.inv(virginica_cov)

#Test 1. Setosa Class
Dist_seto = []
Dist_seto.append(np.matmul(np.matmul(np.transpose(setosa_mean-Test.iloc[0]),setosa_A),setosa_mean-Test.iloc[0]))
Dist_seto.append(np.matmul(np.matmul(np.transpose(versi_mean-Test.iloc[0]),versi_A),versi_mean-Test.iloc[0]))
Dist_seto.append(np.matmul(np.matmul(np.transpose(virginica_mean-Test.iloc[0]),virginica_A),virginica_mean-Test.iloc[0]))

#Test 2. Versicolor Class
Dist_versi = []
Dist_versi.append(np.matmul(np.matmul(np.transpose(setosa_mean-Test.iloc[1]),setosa_A),setosa_mean-Test.iloc[1]))
Dist_versi.append(np.matmul(np.matmul(np.transpose(versi_mean-Test.iloc[1]),versi_A),versi_mean-Test.iloc[1]))
Dist_versi.append(np.matmul(np.matmul(np.transpose(virginica_mean-Test.iloc[1]),virginica_A),virginica_mean-Test.iloc[1]))

#Test 3. Virginica Class
Dist_virgi = []
Dist_virgi.append(np.matmul(np.matmul(np.transpose(setosa_mean-Test.iloc[2]),setosa_A),setosa_mean-Test.iloc[2]))
Dist_virgi.append(np.matmul(np.matmul(np.transpose(versi_mean-Test.iloc[2]),versi_A),versi_mean-Test.iloc[2]))
Dist_virgi.append(np.matmul(np.matmul(np.transpose(virginica_mean-Test.iloc[2]),virginica_A),virginica_mean-Test.iloc[2]))

print(Dist_seto)
print(Dist_versi)
print(Dist_virgi)
Category = ["Setosa", "Versicolor","Virginica"]

# plot with test points
plt.scatter(data_X['petal.length'].iloc[0:49],data_X['petal.width'].iloc[0:49], color='r')
plt.scatter(data_X['petal.length'].iloc[49:98],data_X['petal.width'].iloc[49:98], color='#90ee90')
plt.scatter(data_X['petal.length'].iloc[98:147],data_X['petal.width'].iloc[98:147], color='yellow')
plt.plot(Test.iloc[0,0],Test.iloc[0,1], "d", color='b', markersize=10)
plt.plot(Test.iloc[1,0],Test.iloc[1,1], "d", color='r', markersize=10)
plt.plot(Test.iloc[2,0],Test.iloc[2,1], "d", color='purple', markersize=10)
plt.show()

print("The distance of Row 0 wrt to other classes is {x} and it belongs to {clas} class".format(x=np.around(Dist_seto,3),clas=Category[Dist_seto.index(min(Dist_seto))]))
print("The distance of Row 50 wrt to other classes is {x} and it belongs to {clas} class".format(x=np.around(Dist_versi,3),clas=Category[Dist_versi.index(min(Dist_versi))]))
print("The distance of Row 100 wrt to other classes is {x} and it belongs to {clas} class".format(x=np.around(Dist_virgi,3),clas=Category[Dist_virgi.index(min(Dist_virgi))]))