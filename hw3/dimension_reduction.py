import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings("ignore")


def handle_model(model, X, Y, X_test, Y_test):
    model.fit(X, Y)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction)
    print(f"    {model.__class__.__name__} Accuracy: {accuracy:.2f}")


def GNB(df_train, df_test):
    X = df_train.drop("label", axis=1)
    Y = df_train["label"]
    X_test = df_test.drop("label", axis=1)
    Y_test = df_test["label"]
    handle_model(GaussianNB(), X, Y, X_test, Y_test)


def LGR(df_train, df_test):
    X = df_train.drop("label", axis=1)
    Y = df_train["label"]
    X_test = df_test.drop("label", axis=1)
    Y_test = df_test["label"]
    handle_model(LogisticRegression(max_iter=90), X, Y, X_test, Y_test)


def main():
    df_train = pd.read_csv("fashion-mnist_train.csv")
    df_test = pd.read_csv("fashion-mnist_test.csv")
    train_labels = df_train["label"]
    test_labels = df_test["label"]

    # Original Data
    print("Original:")
    GNB(df_train, df_test)
    LGR(df_train, df_test)

    #
    n_components = 10
    svd = TruncatedSVD(n_components=n_components)

    X_train_svd = svd.fit_transform(df_train.drop("label", axis=1))
    X_test_svd = svd.transform(df_test.drop("label", axis=1))

    df_train_svd = pd.DataFrame(
        data=X_train_svd, columns=[f"svd_{i}" for i in range(n_components)]
    )
    df_test_svd = pd.DataFrame(
        data=X_test_svd, columns=[f"svd_{i}" for i in range(n_components)]
    )
    df_train_svd["label"] = train_labels
    df_test_svd["label"] = test_labels

    print("SVD: ")
    GNB(df_train_svd, df_test_svd)
    LGR(df_train_svd, df_test_svd)


if __name__ == "__main__":
    main()
