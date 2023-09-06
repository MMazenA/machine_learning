import numpy as np
from numpy import *
from matplotlib import pyplot as plt


X = []  
y = []   


def loadDataSet():
    f=open('Question5.txt')
    # Read data line by line and use strip to remove the Spaces 
    for line in f.readlines():
        nline=line.strip().split()
        # X has two columns
        X.append([float(nline[0]),float(nline[1])])
        y.append(int(nline[2]))
    return mat(X).T,mat(y)


X,y=loadDataSet()


def sigmoid(x):
    #==========
    #todo '''complete the sigmoid function'''
    #==========


def Logistic(X,y,W,b,n,alpha,iterations):
    
    '''
    X: input data
    y: labels
    W: weight
    b: bias
    n: number of samples
    alpha: learning rate
    iterations: the number of iteration
    '''
    
    J = zeros((iterations,1))
    for i in range(iterations):   
        
        # step1 forward propagation
        #==========
        #todo '''complete forward propagation equation'''
        #==========
        
        # compute cost function
        #==========
        #todo '''complete compute cost function equation'''
        #==========
        
        # step2 backpropagation
        #==========
        #todo '''complete backpropagation equations'''
        #==========
        
        # step3 gradient descent
        #==========
        #todo '''complete gradientdescent equations'''
        #==========
    return y_hat,W,b,J


def plotBestFit(X,y,J,W,b,n,y_hat):
    
    '''
    X: input data
    y: labels
    J: cost values
    W: weight
    b: bias
    n: number of samples
    y_hat: the predict labels from Logistic Regression 
    '''
    
    # Plot cost function figure
    #==========
    #todo '''complete the code to plot cost function results'''
    #==========
    
    # Plot the final classification figure
    #==========
    #todo '''complete the code to Logistic Regression Classification Result'''
    #==========
    
    plt.show()


num = X.shape[0]  # number of features
n = X.shape[1] # number of samples


# Initianlize the weights and bias
#==========
#todo '''complete the code to initianlize the weights and bias'''
#==========
W = 
b = 


# Learning rate
#==========
#todo '''try different learning rates''
#==========
alpha= 


# Iterations
#==========
#todo '''try different Iterations''
#==========
iterations = 


# Get the results from Logistic function
y_hat,W,b,J = Logistic(X, y, W, b, n, alpha, iterations)


# Plot figures
plotBestFit(X, y, J, W, b, n, y_hat)