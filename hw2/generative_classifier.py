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

    def discrete_probabilities(self):
        """Produce probability chart for each features count with respect to its actual output"""
        dict_training_set = dict(tuple(self.training_set.groupby(self.predicting_feature)))
        col_names = list(self.discrete_cols)
        discrete_probabalities = dict.fromkeys(col_names)
        for key in dict_training_set:
            feature_dict = {}
            for col in col_names:
                feature_dict[col] = (dict_training_set[key][col].value_counts())
            discrete_probabalities[key] = feature_dict.copy()
        return discrete_probabalities
    
    def continous_probabilities(self):
        """Produce probability chart for each features count with respect to its actual output"""
        dict_training_set = dict(tuple(self.training_set.groupby(self.predicting_feature)))
        col_names = list(self.continues_cols)
        continous_probabilities = dict.fromkeys(self.possible_outputs)
        for key in continous_probabilities:
            feature_dict = {}
            for feature in col_names:
                feature_mean = np.mean(dict_training_set[key][feature])
                feature_variance = np.std(dict_training_set[key][feature])
                feature_dict[feature] = [feature_mean,feature_variance]
            continous_probabilities[key] = feature_dict.copy()
        return continous_probabilities
        
    def occurance_probabilities(self):
        value_counts = self.training_set.value_counts(self.predicting_feature)
        return value_counts
    
    def probability_table(self):
        cont = self.continous_probabilities()
        disc = self.discrete_probabilities()
        for key in cont:
            cont[key].update(disc[key])
        return cont
    
    def train(self):
        self.split_data()
        self.probabilities = self.probability_table()
        self.occurance_prob = self.occurance_probabilities()

    def predict_single(self,row: pd.DataFrame):
        sol = {} #dict of size class_count for all the chances of the test case
        for key in self.possible_outputs:
            # print(self.occurance_prob[key],end=" ")
            likelihood = np.log(self.occurance_prob[key])
            for feature in row.columns:
                if feature in self.continues_cols:
                    #z score
                    mstd = self.probabilities[key][feature]
                    mean = mstd[0]
                    std = mstd[1]
                    likelihood+=np.log((row[feature] - mean)/std)
                    pass
                else:
                    row_feature = row[feature]
                    feature_occurance = self.probabilities[key][feature][row_feature].iloc[0]
                    print(feature_occurance,row_feature)
                    feature_total = sum(self.probabilities[key][feature])
                    likelihood+= np.log(feature_occurance/feature_total)
                    
                    
                    
            print(likelihood)
            sol[key] = likelihood

            
    def test(self):

        self.predict_single(self.testing_set.head(1))



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
x.test()

# df = (data[["age"]] - data[["age"]].mean())/data[["age"]].std() 

# print(df.tail(5))
# print(df[["age"]].mean())
# print("Wjat",(53-data[["age"]].mean())/data[["age"]].std())
