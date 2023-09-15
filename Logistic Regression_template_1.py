import numpy as np
from numpy import *
from matplotlib import pyplot as plt


X = []
y = []


def loadDataSet():
    f = open("Question5.txt")
    # Read data line by line and use strip to remove the Spaces
    for line in f.readlines():
        nline = line.strip().split()
        # X has two columns
        X.append([float(nline[0]), float(nline[1])])
        y.append(int(nline[2]))
    return mat(X).T, mat(y)


# features are seperated in X[feature_number][row_number]
X, y = loadDataSet()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_single_value(weight, row, bias):
    row = np.array(row)
    feature_count = len(row)
    logit = bias
    for i in range(feature_count):
        logit += weight[i] * row[i]
    predicted_value = sigmoid(logit)

    return predicted_value


def predict_all_values(X, weight, bias):
    feature_count, row_count = X.shape
    predicted_values = []

    for i in range(row_count):
        predicted_value = predict_single_value(weight, X[:, i], bias)
        predicted_values.append(predicted_value)
    return np.array(predicted_values)


def compute_loss(y, y_hat):

    pass


def Logistic(X, y, W, b, n, alpha, iterations):

    """
    X: input data
    y: labels
    W: weight
    b: bias
    n: number of samples
    alpha: learning rate
    iterations: the number of iteration
    """
    # y = np.array(y)
    # y = y.reshape(-1)
    # print("Y EQUALS", y.shape)
    J = zeros((iterations, 1))
    for i in range(iterations):

        # step1 forward propagation
        # ==========
        # todo '''complete forward propagation equation'''
        # ==========
        y_hat = predict_all_values(X, W, b)
        print(y_hat)

        # compute cost function
        # ==========
        # todo '''complete compute cost function equation'''
        # ==========
        J[i] = np.sum()

        # step2 backpropagation
        # ==========
        # todo '''complete backpropagation equations'''
        # ==========
        dz = y_hat - y
        dW = (1 / n) * np.dot(X, dz.T)
        db = (1 / n) * np.sum(dz)

        # step3 gradient descent
        # ==========
        # todo '''complete gradientdescent equations'''
        # ==========
        W = W - alpha * dW
        b = b - alpha * db

    return y_hat, W, b, J


# def plotBestFit(X, y, J, W, b, n, y_hat):

#     """
#     X: input data
#     y: labels
#     J: cost values
#     W: weight
#     b: bias
#     n: number of samples
#     y_hat: the predict labels from Logistic Regression
#     """

#     # Plot cost function figure
#     # ==========
#     # todo '''complete the code to plot cost function results'''
#     # ==========

#     # Plot the final classification figure
#     # ==========
#     # todo '''complete the code to Logistic Regression Classification Result'''
#     # ==========

#     plt.show()


# num = X.shape[0]  # number of features
# n = X.shape[1] # number of samples


# Initianlize the weights and bias
# ==========
# todo '''complete the code to initianlize the weights and bias'''
# ==========
W = np.arange(0.01, 0.11, 0.001)
b = 0.05
# Learning rate
# ==========
# todo '''try different learning rates''
# ==========
alpha = 0.01


# Iterations
# ==========
# todo '''try different Iterations''
# ==========
iterations = 1000


# Get the results from Logistic function
y_hat, W, b, J = Logistic(X, y, W, b, X.shape[1], alpha, iterations)
print(J)


# Plot figures
# plotBestFit(X, y, J, W, b, n, y_hat)
