"""Mazen Mirza"""
from statistics import NormalDist
import numpy as np
import pandas as pd

class knn():
    def __init__(self, dataset: pd.DataFrame, predicting_feature: str, k = 1) -> None:
        self.raw_dataset = dataset
        self.dataset = dataset
        self.testing_set = None
        self.training_set = None
        self.predicting_feature = predicting_feature
        self.k=k
        self.confusion_matrix = None
        
        # map boolean features to strings
        mask = self.dataset.map(type) != bool
        replace = {True: "TRUE", False: "FALSE"}
        self.dataset = self.dataset.where(mask, self.dataset.replace(replace))


    def split_data(self, test_ratio=0.2):
        """Split the dataset into training and testing sets."""
        testing_size = int(self.dataset.shape[0] * test_ratio)
        self.testing_set = self.dataset.tail(testing_size)
        self.training_set = self.dataset.drop(self.testing_set.index)
        self.possible_outputs = self.dataset[self.predicting_feature].unique()
    
    def onehot_encode(self, columns_to_encode):
            """Perform one-hot encoding for specified columns in the dataset."""
            # Create a copy of the dataset to avoid modifying it
            encoded_dataset = self.dataset.copy()

            for column in columns_to_encode:
                if column in encoded_dataset.columns:
                    # Use pandas' factorize function to label encode the column
                    encoded_dataset[column] = pd.factorize(encoded_dataset[column])[0]
            self.dataset = encoded_dataset
    def mode_replacement(self):
        for column_name in self.dataset.columns:
            mode_value = self.dataset[column_name].mode().values[0]
            self.dataset[column_name].fillna(mode_value, inplace=True)

    def calculate_distance(self, point1, point2):
        """Calculate the Euclidean distance between two data points."""

        features1 = point1.values[0][:-1]  # Extract features from the first point
        features2 = point2.values[0][:-1]  # Extract features from the second point
        squared_diff = np.sum((features1 - features2) ** 2,axis=0)
        euclidean_distance = np.sqrt(squared_diff)
        return euclidean_distance
    
    def predict_single(self, input_point):
        """Predict the class for the input data point."""
        distances = [self.calculate_distance(input_point, self.training_set.iloc[[i]]) for i in range(self.training_set.shape[0])]
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_classes = [self.training_set.iloc[i][self.predicting_feature] for i in nearest_indices]
        predicted_class = max(nearest_classes, key=nearest_classes.count)
        return predicted_class

    def test(self, testing_set=None):
        """Produce a confusion matrix from the testing set."""
        if testing_set:
            self.testing_set = testing_set
        row_count = self.testing_set.shape[0]
        confusion_matrix = {
            "true_positive": 0,
            "false_positive": 0,
            "true_negative": 0,
            "false_negative": 0,
        }
        for i in range(row_count):
            input_point = self.testing_set.iloc[[i]]
            prediction = self.predict_single(input_point)  # Use the predict method to get the prediction
            true_value = input_point[self.predicting_feature].values[0]
            if prediction == 0:
                # Prediction is negative
                if true_value != 0:
                    confusion_matrix["false_negative"] += 1
                else:
                    confusion_matrix["true_negative"] += 1
            else:
                # Prediction is positive
                if true_value != 1:
                    confusion_matrix["false_positive"] += 1
                else:
                    confusion_matrix["true_positive"] += 1
        self.confusion_matrix = confusion_matrix
        return confusion_matrix
    
    def testing_results(self):
        if self.confusion_matrix == None:
            Exception("Need to compute .test() before showing results")
        #true and falses
        tp = self.confusion_matrix["true_positive"]
        tn = self.confusion_matrix["true_negative"]
        fp = self.confusion_matrix["false_positive"]
        fn = self.confusion_matrix["false_negative"]
       
        pr = tp/(tp+fp)
        re = tp/(tp+fn)
        ca = (tp+tn)/(tp+tn+fp+fn)
        f1 = (2*tp)/(2*tp+fp+fn)

        print(self.confusion_matrix)
        print(f"Precision:{pr*100:8.3f}%")
        print(f"Recall:{re*100:11.3f}%")
        print(f"Accuracy:{ca*100:>9.3f}%")
        print(f"F1:{f1*100:15.3f}%")
        return [pr,re,ca,f1]




def main():
    """Example usage."""
    data = pd.read_csv("heart_disease_uci.csv")
    data.drop(["id"], axis=1, inplace=True)
    data["num"] = data["num"].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
    # data = data.sample(frac=1)

    knn_test = knn(data, predicting_feature="num",k=9)
    knn_test.onehot_encode(["age","sex","dataset","cp","fbs","restecg","exang","slope","thal"])
    knn_test.split_data()
    knn_test.mode_replacement()
    knn_test.test()
    knn_test.testing_results()


if __name__ == "__main__":
    main()
