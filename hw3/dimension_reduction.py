import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")


def handle_model(model, df_train, df_test):

    X = df_train.drop("label", axis=1)
    Y = df_train["label"]
    X_test = df_test.drop("label", axis=1)
    Y_test = df_test["label"]
    model = model
    model.fit(X, Y)
    prediction = model.predict(X_test)

    accuracy = accuracy_score(Y_test, prediction)
    print(f"Accuracy: {accuracy:.2f}")


def GNB(df_train, df_test):
    handle_model(GaussianNB(), df_train, df_test)


def LGR(df_train, df_test):
    handle_model(LogisticRegression(max_iter=90), df_train, df_test)


def main():
    df_train = pd.read_csv("fashion-mnist_train.csv")
    df_test = pd.read_csv("fashion-mnist_test.csv")
    GNB(df_train, df_test)
    LGR(df_train, df_test)


if __name__ == "__main__":
    main()
