import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Fashion-MNIST Dataset
train_data = pd.read_csv("fashion-mnist_train.csv")
X_train = train_data.drop("label", axis=1)
y_train = train_data["label"]

test_data = pd.read_csv("fashion-mnist_test.csv")
X_test = test_data.drop("label", axis=1)
y_test = test_data["label"]

# Perform SVD Dimension Reduction
svd = TruncatedSVD(n_components=10)
X_train_svd = svd.fit_transform(X_train)

# Train Classifiers
nb_classifier = MultinomialNB()
knn_classifier = KNeighborsClassifier(n_neighbors=5)
mlr_classifier = LogisticRegression(max_iter=90)

nb_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
mlr_classifier.fit(X_train, y_train)

nb_classifier_svd = MultinomialNB()
knn_classifier_svd = KNeighborsClassifier(n_neighbors=5)
mlr_classifier_svd = LogisticRegression(max_iter=90)

nb_classifier_svd.fit(X_train_svd, y_train)
knn_classifier_svd.fit(X_train_svd, y_train)
mlr_classifier_svd.fit(X_train_svd, y_train)

y_pred_nb = nb_classifier.predict(X_test)
y_pred_knn = knn_classifier.predict(X_test)
y_pred_mlr = mlr_classifier.predict(X_test)

y_pred_nb_svd = nb_classifier_svd.predict(svd.transform(scaler.transform(X_test)))
y_pred_knn_svd = knn_classifier_svd.predict(svd.transform(scaler.transform(X_test)))
y_pred_mlr_svd = mlr_classifier_svd.predict(svd.transform(scaler.transform(X_test)))

accuracy_original = (
    accuracy_score(y_test, y_pred_nb),
    accuracy_score(y_test, y_pred_knn),
    accuracy_score(y_test, y_pred_mlr),
)
accuracy_svd = (
    accuracy_score(y_test, y_pred_nb_svd),
    accuracy_score(y_test, y_pred_knn_svd),
    accuracy_score(y_test, y_pred_mlr_svd),
)

print("Original Data Accuracy (NB, KNN, MLR):", accuracy_original)
print("SVD-Reduced Data Accuracy (NB, KNN, MLR):", accuracy_svd)
