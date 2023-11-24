import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
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


def KNN(df_train, df_test):
    X = df_train.drop("label", axis=1)
    Y = df_train["label"]
    X_test = df_test.drop("label", axis=1)
    Y_test = df_test["label"]
    handle_model(KNeighborsClassifier(3), X, Y, X_test, Y_test)


def scale_dataset(df):
    labels = df["label"]
    df = df.drop("label", axis=1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    scaled_df["label"] = labels

    return scaled_df


def main():
    df_train = pd.read_csv("fashion-mnist_train.csv")
    df_test = pd.read_csv("fashion-mnist_test.csv")
    df_train = scale_dataset(df_train)
    df_test = scale_dataset(df_test)
    train_labels = df_train["label"]
    test_labels = df_test["label"]

    df_train.drop("label", axis=1, inplace=True)
    df_test.drop("label", axis=1, inplace=True)

    df_train = pd.DataFrame(
        MinMaxScaler().fit_transform(df_train), columns=df_train.columns
    )
    df_test = pd.DataFrame(
        MinMaxScaler().fit_transform(df_test), columns=df_test.columns
    )

    df_train["label"] = train_labels
    df_test["label"] = test_labels

    print("Original:")
    GNB(df_train, df_test)
    LGR(df_train, df_test)
    KNN(df_train, df_test)

    features = 11
    svd = TruncatedSVD(n_components=features)

    X_train_svd = svd.fit_transform(df_train.drop("label", axis=1))
    X_test_svd = svd.transform(df_test.drop("label", axis=1))

    df_train_svd = pd.DataFrame(
        data=X_train_svd, columns=[f"svd_{i}" for i in range(features)]
    )
    df_test_svd = pd.DataFrame(
        data=X_test_svd, columns=[f"svd_{i}" for i in range(features)]
    )
    df_train_svd["label"] = train_labels
    df_test_svd["label"] = test_labels

    print("SVD: ")
    GNB(df_train_svd, df_test_svd)
    LGR(df_train_svd, df_test_svd)
    KNN(df_train_svd, df_test_svd)


if __name__ == "__main__":
    main()
