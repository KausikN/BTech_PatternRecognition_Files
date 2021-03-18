'''
Q6. Consider the 128- dimensional feature vectors given in the “face feature vectors.csv” file.
Use this information to design and implement a Bayes Classifier.
Dataset Specifications:
Total number of samples = 800
Number of classes = 2 ( labelled as “male” and “female”)
Samples from “1 to 400” belongs to class “male”
Samples from “401 to 800” belongs to class “female”
Number of samples per class = 400
Use the following information to design classifier:
    Number of test feature vectors ( first 5 in each class) = 5
    Number of training feature vectors ( remaining 395 in each class) = 395
    Number of dimensions = 128
'''

# Imports
from csv import reader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, exp
import random
import sympy
from sympy import symbols, plot_implicit
from Utils import *

# Main Functions
# load dataset from csv file
def load_csv(filename):
    # import data to "dataset" variable
    dataset = list()
    with open(filename, 'r') as file:
        csv_data = reader(file)
        for row in csv_data:
            if not row:
                continue
            row.append(row[1])
            dataset.append(row[2:])
        dataset = dataset[1:]
    # convert the values to float from string
    for column in range(len(dataset[0])-1):
        for row in dataset:
            row[column] = float(row[column].strip())
    
    # convert the class column to integer
    class_values = [row[len(dataset[0])-1] for row in dataset]
    unique_class_values = set(class_values)
    class_values_lookup_table = dict()
    for i, value in enumerate(unique_class_values):
        class_values_lookup_table[value] = i
    for row in dataset:
        row[len(dataset[0])-1] = class_values_lookup_table[row[len(dataset[0])-1]]

    return dataset 

# separate the dataset values by class values
def dataset_separation(dataset):
    data_separated = dict()
    for i in range(len(dataset)):
        feature_vector = dataset[i]
        class_value = feature_vector[-1]
        if class_value not in data_separated:
            data_separated[class_value] = list()
        data_separated[class_value].append(feature_vector)
    return data_separated

# compute mean of the given input list
def mean(input_list):
    mean = float(sum(input_list) / float(len(input_list)))
    return mean

# compute standard deviation of the given list
def standard_deviation(input_list):
    average = mean(input_list)
    variance = sum([(x-average)**2 for x in input_list]) / float(len(input_list)-1)
    standard_deviation = sqrt(variance)
    return standard_deviation

# compute summary of data by mean, standard deviation and count for each column in dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), standard_deviation(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries

# compute summary of data by mean, standard deviation and count classwise
def summarize_dataset_by_class(dataset):
    class_separated_data = dataset_separation(dataset)
    summaries = dict()
    for class_value, rows in class_separated_data.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

# compute multivariate normal distribution probability distribution function for x
def compute_multivariate_normal_distribution_probability(X, mean, covariance_matrix):
    X_mean = np.array(X)-np.array(mean)
    exponent = exp(-(np.linalg.solve(covariance_matrix, X_mean).T.dot(X_mean)) / 2.0)
    return (1.0 / (np.sqrt(((2 * np.pi)**len(X)) * np.linalg.det(covariance_matrix))) * exponent)

# compute class probabilities for each row of data
def compute_class_probabilities(summaries, row, covariance_matrix):
    total_rows = sum([summaries[label][0][-1] for label in summaries])
    class_probabilities = dict()
    for class_value, class_summaries in summaries.items():
        # apriori probability
        class_probabilities[class_value] = summaries[class_value][0][-1] / float(total_rows)
        mean_vector = []
        for i in range(len(class_summaries)):
            mean, standard_deviation, _ = class_summaries[i]
            mean_vector.append(mean)
        class_probabilities[class_value] *= compute_multivariate_normal_distribution_probability(row[0:-1], mean_vector, covariance_matrix)
    return class_probabilities

# compute accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# predict the class for a given row data
def predict(summaries, row, covariance_matrix):
    class_probabilities = compute_class_probabilities(summaries, row, covariance_matrix)
    best_label, best_probability = None, -1
    for class_value, probability in class_probabilities.items():
        if best_label is None or probability > best_probability:
            best_probability = probability
            best_label = class_value
    return best_label

# bayesian classifier algorithm with scores
def bayes_classifier(train, test, covariance_matrix):
    summarize = summarize_dataset_by_class(train)
    predicted = list()
    scores = list()
    for row in test:
        output = predict(summarize, row, covariance_matrix)
        predicted.append(output)
    actual = [row[-1] for row in test]
    scores = accuracy_metric(actual, predicted)
    return predicted, scores

# dataset file input path
dataset = 'Assignment2/Data/face feature vectors.csv'
# 0 - male
# 1 - female
data = load_csv(dataset)
# Uncomment the below time to scale the data points by 10 power 5
# data = list(np.array(data)*10000)

training_data = data[5:400]+data[405:]
test_data = data[0:5]+data[400:405]

# covariance matrix
data_without_class = []
for row in data:
    data_without_class.append(row[0:-1])
covariance_matrix = np.cov(np.transpose(data_without_class))

training_data_summary = summarize_dataset_by_class(training_data)
predictions, scores = bayes_classifier(training_data, test_data, covariance_matrix)

print("The accuracy is  is {x}%".format(x=scores))
print("The Determinent of Covariance Matrix = ", np.linalg.det(covariance_matrix))
# Disc Func and Decision Boundary
data_pd = data.copy()

data_male = data_pd[0:399]
data_female = data_pd[399:]

# Disc Func
Male_eq = DiscriminantFunctionEquation(data_male, 1/2)
Female_eq = DiscriminantFunctionEquation(data_female, 1/2)
Male_eqpoly = sympy.poly(Male_eq)
Female_eqpoly = sympy.poly(Female_eq)
print("Male Discriminant Func:\n", Male_eqpoly)
print()
print("Female Discriminant Func:\n", Female_eqpoly)
print()

# Decision Boundary
DB_Male_Female = Male_eq - Female_eq

DB_Male_Femalepoly = sympy.poly(DB_Male_Female)
print("Decision Boundary Male vs Female:\n", DB_Male_Femalepoly, "= 0")
print()