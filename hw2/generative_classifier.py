import numpy as np
import pandas as pd
class naive_bayes():
    def __init__(self, dataset : pd.DataFrame, predicting_feature: str, continues_cols = None) -> None:
        """Inits naive bayes class using pandas dataframe.Normalizes based on Z-score."""
        self.dataset = dataset
        self.testing_set = None
        self.training_set = None
        self.predicting_feature = predicting_feature
        self.continues_cols = None
        
        
    def split_data(self, test_ratio=0.2):
        """Split the dataset into training and testing sets."""
        testing_size = int(self.dataset.shape[0] * test_ratio)
        self.testing_set = self.dataset.tail(testing_size)
        self.training_set = self.dataset.drop(self.testing_set.index)
        self.possible_outputs = self.dataset[self.predicting_feature].unique()


    def train(self):
        self.split_data()
        dict_training_set = dict(tuple(self.training_set.groupby(self.predicting_feature)))
        for key in dict_training_set:
            group_total = dict_training_set[key].shape[0]

    def gaussian_likelihood(self,x,mean,std):
                

    def normalize(self,continues_cols):
        """Normalize columns that are continous using Z score."""
        self.continues_cols = continues_cols
        removing = set(continues_cols).union(set(self.predicting_feature))
        self.discrete_cols = set(self.dataset.columns) - removing
        normalized_set = self.dataset.copy()
        normalized_set[continues_cols] = (
            self.dataset[continues_cols] - self.dataset[continues_cols].mean()
        ) / self.dataset[continues_cols].std()
        self.dataset = normalized_set

class knn():
    def innit():
        pass




data = pd.read_csv("heart_disease_uci.csv")
data.drop(["id"],axis=1,inplace=True)
x = naive_bayes(data,"num")
x.normalize(["age", "trestbps", "chol", "thalch", "oldpeak"])
x.train()