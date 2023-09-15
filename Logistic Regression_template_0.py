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
    return mat(X).T, np.array(y)


# features are seperated in X[feature_number][row_number]
X, y = loadDataSet()


def compute_loss(y, y_hat):

    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def predict(x, w, bias):
    """Weight * value for every feature in a given sample, for every sample"""
    num = x.shape[0]  # number of features
    n = x.shape[1]  # number of samples
    prediction = []
    for sample in range(n):
        logit = bias
        for feature in range(num):
            logit += x[feature, sample] * w[feature]
        prediction.append((logit))
    return sigmoid(np.array(prediction))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# W = np.array([0.01, 0.02])
# b = 0
# alpha = 0.01
# iterations = 1000
# y_hat = predict(X, W, b)
# print(y.shape, y_hat.shape)
# compute_loss(y, y_hat)
# dW = np.dot(X, (y_hat - y).T)
# dB = np.sum(y_hat - y)
# print(dW)
# print(dB)


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

    J = zeros((iterations, 1))
    for i in range(iterations):

        # step1 forward propagation
        # ==========
        # todo '''complete forward propagation equation'''
        # ==========
        y_hat = predict(X, W, b)
        # compute cost function
        # ==========
        # todo '''complete compute cost function equation'''
        # ==========
        J[i] = compute_loss(y, y_hat)
        # step2 backpropagation
        # ==========
        # todo '''complete backpropagation equations'''
        # ==========

        dW = np.dot(X, (y_hat - y).T)
        dB = np.sum(y_hat - y)

        # step3 gradient descent
        # ==========
        # todo '''complete gradientdescent equations'''
        # ==========
        W = W - alpha * dW
        b = b - alpha * dB

    return y_hat, W, b, J


# def plotBestFit(X,y,J,W,b,n,y_hat):

#     '''
#     X: input data
#     y: labels
#     J: cost values
#     W: weight
#     b: bias
#     n: number of samples
#     y_hat: the predict labels from Logistic Regression
#     '''

#     # Plot cost function figure
#     #==========
#     #todo '''complete the code to plot cost function results'''
#     #==========

#     # Plot the final classification figure
#     #==========
#     #todo '''complete the code to Logistic Regression Classification Result'''
#     #==========

#     plt.show()


# num = X.shape[0]  # number of features
# n = X.shape[1] # number of samples


W = np.array([0.01, 0.02])
b = 0
alpha = 0.01
iterations = 100


# Get the results from Logistic function
y_hat, W, b, J = Logistic(X, y, W, b, X.shape[1], alpha, iterations)
print(J[-1])


# # Plot figures
# plotBestFit(X, y, J, W, b, n, y_hat)
