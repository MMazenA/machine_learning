import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    X = []
    y = []
    with open("Question5.txt", "r") as f:
        for line in f.readlines():
            nline = line.strip().split()
            X.append([float(nline[0]), float(nline[1])])
            y.append(int(nline[2]))
    return np.array(X).T, np.array(y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Logistic regression
def Logistic(X, y, W, b, n, alpha, iterations):
    J = np.zeros((iterations, 1))

    for i in range(iterations):
        # Step 1: Forward propagation

        z = np.dot(W.T, X) + b
        # print(z.shape)
        y_hat = sigmoid(z)
        # calculate loss
        J[i] = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        # Step 2: Backpropagation

        # not sure which one is correct
        # dW = (1 / n) * np.dot(X, dz.T)
        # db = (1 / n) * np.sum(dz)
        dW = np.dot(X, (y_hat - y).T)
        db = np.sum(y_hat - y)
        # print(dW)

        # Step 3: Gradient descent
        W = W - alpha * dW
        b = b - alpha * db

    return y_hat, W, b, J


def plotBestFit(X, y, J, W, b, n, y_hat):
    # plot cost
    plt.figure()
    plt.plot(range(iterations), J)
    plt.xlabel("Iterations")
    plt.ylabel("Cost Function")

    # plot scatter with different colored 0/1
    plt.figure()
    # feature1,feature2, colored by 0/1
    feature1_values = X[0, :]
    feature2_values = X[1, :]
    plt.scatter(feature1_values, feature2_values, c=y)
    low = min(feature2_values) - 1
    high = max(feature1_values) + 1
    inputs = np.linspace(low, high, 100)
    # w^t * X + b = 0
    line = (-W[0] / W[1]) * inputs - (b / W[1])
    plt.plot(inputs, line, color="black", linestyle="dashed")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Logistic Regression Decision Boundary")
    plt.show()
    plt.show()


X, y = loadDataSet()
num = X.shape[0]
n = X.shape[1]
W = np.array([0.01, 0.02])
b = -1
alpha = 0.01
iterations = 400
y_hat, W, b, J = Logistic(X, y, W, b, n, alpha, iterations)
print(W, J[-1])
plotBestFit(X, y, J, W, b, n, y_hat)
