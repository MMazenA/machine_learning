import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# for tree
class Node:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.prediction = None


def calculate_entropy(y):
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def find_best_split(X, y):
    m, n = X.shape
    if m <= 1:
        return None, None  # if there's one sample

    parent_entropy = calculate_entropy(y)

    best_info_gain = 0
    best_index = None
    best_threshold = None

    for i in range(n):
        thresholds = np.unique(X[:, i])
        for threshold in thresholds:
            left_mask = X[:, i] <= threshold
            right_mask = [not value for value in left_mask]  # flipping all bools

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_entropy = calculate_entropy(y[left_mask])
            right_entropy = calculate_entropy(y[right_mask])

            weighted_entropy = (np.sum(left_mask) / m) * left_entropy + (
                np.sum(right_mask) / m
            ) * right_entropy

            info_gain = parent_entropy - weighted_entropy

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_index = i
                best_threshold = threshold

    return best_index, best_threshold


def build_tree(X, y, cutoff, depth=1, max_depth=None):
    if depth == max_depth or len(np.unique(y)) == 1 or calculate_entropy(y) < cutoff:
        node = Node(data=None, target=y)
        node.prediction = np.argmax(np.bincount(y))
        return node

    index, threshold = find_best_split(X, y)

    if index is None:
        node = Node(data=None, target=y)
        node.prediction = np.argmax(np.bincount(y))
        return node

    left_mask = X[:, index] <= threshold
    right_mask = [not value for value in left_mask]

    node = Node(data=(index, threshold), target=y)

    node.left = build_tree(X[left_mask], y[left_mask], cutoff, depth + 1, max_depth)
    node.right = build_tree(X[right_mask], y[right_mask], cutoff, depth + 1, max_depth)

    return node


def predict_tree(node, X):
    if node.left is None and node.right is None:
        return node.prediction

    index, threshold = node.data

    if X[index] <= threshold:
        return predict_tree(node.left, X)
    else:
        return predict_tree(node.right, X)


def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def plot_errors(theta_list, train_error_list, val_error_list):
    plt.plot(theta_list, train_error_list, label="Training Error")
    plt.plot(theta_list, val_error_list, label="Validation Error")
    plt.xlabel("Theta")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


df = pd.read_csv("adult.csv")

X = df.drop(["income"], axis=1).to_numpy()
y = (df["income"] == ">50K").astype(int).to_numpy()


# random state is shuffling according to a key
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=0
)

# user can stop recursive splitting here or in max_depth
theta_list = [0.2, 0.4, 0.6, 0.8]
train_error_list = []
val_error_list = []

for theta in theta_list:
    tree = build_tree(X_train, y_train, theta, max_depth=None)

    y_pred_train = np.array([predict_tree(tree, x) for x in X_train])
    train_error = 1 - calculate_accuracy(y_train, y_pred_train)
    train_error_list.append(train_error)

    y_pred_val = np.array([predict_tree(tree, x) for x in X_val])
    val_error = 1 - calculate_accuracy(y_val, y_pred_val)
    val_error_list.append(val_error)

plot_errors(theta_list, train_error_list, val_error_list)
