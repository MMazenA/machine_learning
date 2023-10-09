from statistics import NormalDist
import numpy as np
import pandas as pd


class NaiveBayes:
    """Naive bayes classifier, requires python3.10"""

    def __init__(
        self, dataset: pd.DataFrame, predicting_feature: str, continues_cols=None
    ) -> None:
        """Inits naive bayes class using pandas dataframe.Normalizes based on Z-score."""
        self.raw_dataset = dataset
        self.dataset = dataset
        self.testing_set = None
        self.training_set = None

        self.predicting_feature = predicting_feature
        self.continues_cols = continues_cols
        self.possible_outputs = None
        self.discrete_cols = None
        self.probabilities = None
        self.occurance_prob = None
        self.confusion_matrix = None

        # map boolean features to strings
        mask = self.dataset.map(type) != bool
        replace = {True: "TRUE", False: "FALSE"}
        self.dataset = self.dataset.where(mask, self.dataset.replace(replace))
    def mode_replacement(self):
        for column_name in self.dataset.columns:
            mode_value = self.dataset[column_name].mode().values[0]
            self.dataset[column_name].fillna(mode_value, inplace=True)

    def split_data(self, test_ratio=0.2):
        """Split the dataset into training and testing sets."""
        testing_size = int(self.dataset.shape[0] * test_ratio)
        self.testing_set = self.dataset.tail(testing_size)
        self.training_set = self.dataset.drop(self.testing_set.index)
        self.possible_outputs = self.dataset[self.predicting_feature].unique()

    def normalize(self, continues_cols):
        """Normalize columns that are continous using Z score."""
        self.continues_cols = continues_cols
        removing = set(continues_cols).union(set(self.predicting_feature))
        self.discrete_cols = set(self.dataset.columns) - removing
        normalized_set = self.dataset.copy()
        normalized_set[continues_cols] = (
            self.dataset[continues_cols] - self.dataset[continues_cols].mean()
        ) / self.dataset[continues_cols].std()
        self.dataset = normalized_set

    def discrete_probabilities(self):
        """Produce probability chart for each features count with respect to its actual output"""
        dict_training_set = dict(
            tuple(self.training_set.groupby(self.predicting_feature))
        )
        col_names = list(self.discrete_cols)
        discrete_probabalities = dict.fromkeys(col_names)
        for key in dict_training_set:
            feature_dict = {}
            for col in col_names:
                feature_dict[col] = dict_training_set[key][col].value_counts()
            discrete_probabalities[key] = feature_dict.copy()
        return discrete_probabalities

    def continous_probabilities(self):
        """
        Produce every features mean and variance for continous features.
        Assumes variance is equal between classes.
        """
        dict_training_set = dict(
            tuple(self.training_set.groupby(self.predicting_feature))
        )
        col_names = list(self.continues_cols)
        continous_probabilities = dict.fromkeys(self.possible_outputs)
        for key in continous_probabilities:
            feature_dict = {}
            for feature in col_names:
                feature_mean = np.mean(dict_training_set[key][feature])
                feature_variance = np.std(dict_training_set[key][feature])
                feature_dict[feature] = [feature_mean, feature_variance]
            continous_probabilities[key] = feature_dict.copy()
        return continous_probabilities

    def occurance_probabilities(self):
        """Computes the probability that a class occurs."""
        value_counts = self.training_set.value_counts(self.predicting_feature)
        return value_counts

    def probability_table(self):
        """Computes discrete and continous probabilities into one table."""
        cont = self.continous_probabilities()
        disc = self.discrete_probabilities()
        for key in cont:
            cont[key].update(disc[key])
        return cont

    def train(self):
        """Splits, produces probability tables, and produces mean and variance."""
        self.split_data()
        self.probabilities = self.probability_table()
        self.occurance_prob = self.occurance_probabilities()

    def predict_single(self, row: pd.DataFrame, alpha, feature_count):
        """Produces prediction for a single row."""
        sol = {}
        row_without_classifier = row.drop([self.predicting_feature], axis=1)
        for key in self.possible_outputs:
            likelihood = np.log(self.occurance_prob[key] / sum(self.occurance_prob))

            for feature in row_without_classifier.columns:
                if feature in self.continues_cols:
                    mstd = self.probabilities[key][feature]
                    mean = mstd[0]
                    std = mstd[1]
                    z_score = (row[feature].values - mean) / std
                    if z_score > 0:
                        z_score = -z_score
                    cdf = NormalDist().cdf(z_score) * 2
                    likelihood += np.log(cdf)
                else:
                    row_feature = row[feature]
                    seen = row_feature.values[0] in self.probabilities[key][feature]
                    if not seen:
                        feature_occurance = 0
                        feature_total = 0
                    else:
                        feature_occurance = self.probabilities[key][feature][
                            row_feature.values
                        ].values
                        feature_total = sum(self.probabilities[key][feature])
                    probability = (feature_occurance + alpha) / (
                        feature_total + (alpha * feature_count)
                    )
                    likelihood += np.log(probability)
            sol[key] = likelihood

        maximum = ["key", float("-inf")]
        for key, value in sol.items():
            if value > maximum[1]:
                maximum = [key, value]
        return maximum[0]

    def test(self, testing_set=None, alpha=1):
        """Produce confusion matrix from testing set."""
        feature_count = len(self.discrete_cols)
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
            prediction = self.predict_single(
                self.testing_set.iloc[[i]], alpha, feature_count
            )
            true_value = self.testing_set.iloc[[i]][self.predicting_feature].values
            if prediction == 0:
                # predication is negative
                if true_value != 0:
                    confusion_matrix["false_negative"] += 1
                else:
                    confusion_matrix["true_negative"] += 1
            else:
                # predication is postive
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

def cleaned_dataset():
    """Pre-Cleaned Dataset"""
    data = pd.read_csv("heart.csv")
    data = data.sample(frac=1)
    naive_b = NaiveBayes(data, "HeartDisease")
    naive_b.normalize(["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"])
    naive_b.train()
    confusion = naive_b.test(alpha=1)
    print(confusion)



def main():
    """Example usage."""
    data = pd.read_csv("heart_disease_uci.csv")
    data.drop(["id"], axis=1, inplace=True)
    data["num"] = data["num"].map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
    # randomize data
    # data = data.sample(frac=1)
    naive_b = NaiveBayes(data, "num")
    naive_b.normalize(["age", "trestbps", "chol", "thalch", "oldpeak"])
    naive_b.mode_replacement()
    naive_b.train()
    naive_b.test(alpha=1)
    naive_b.testing_results()



if __name__ == "__main__":
    main()
