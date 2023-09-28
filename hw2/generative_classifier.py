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
        mask = self.dataset.applymap(type) != bool
        replace = {True: 'TRUE', False: 'FALSE'}
        self.dataset = self.dataset.where(mask, self.dataset.replace(replace))
        
        
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

        row_without_classifier = row.drop([self.predicting_feature],axis=1)
        print(row)
        
        for key in self.possible_outputs:
            likelihood = self.occurance_prob[key] / sum(self.occurance_prob)
            # print()
            # print()
            # print("liklihood=",likelihood)
            for feature in row_without_classifier.columns:
                    if feature in self.continues_cols and not row[feature].isnull().values.any():
                        #z score
                        mstd = self.probabilities[key][feature]
                        mean = mstd[0]
                        std = mstd[1]
                        z_score = (row[feature].values - mean)/std
                        z_score = 1/abs(z_score)
                        # print("z=",z_score)
                        likelihood+=z_score
                        
                    elif not row[feature].isnull().values.any():
                        row_feature = row[feature]
                        # print(self.probabilities[key][feature][row_feature])
                        try:
                            feature_occurance = self.probabilities[key][feature][row_feature.values].values
                            feature_total = sum(self.probabilities[key][feature])
                        except:
                            feature_occurance = 0 
                            feature_total = 1

                        # print("occurance= ",feature_occurance, " sum=",feature_total)
                        probability = feature_occurance/feature_total
                        likelihood+= probability
                # except:
                #     print("ERROR!!!")
                #     print(row[feature].values)
                #     return
                    # print(feature)
                    # print()
                    # print()
                    
                    
                    
            # print(likelihood)
            sol[key] = likelihood[0]
        maximum =["key",float("-inf")]
        for key in sol:
            if sol[key] > maximum[1]:
                maximum = [key,sol[key]]
        print(maximum)
        print(sol)

            
    def test(self):
        # print(self.testing_set.head(1))
        self.predict_single(self.training_set.iloc[[66]])



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






data = pd.read_csv("heart_disease_uci.csv").dropna()
# print(data.shape)
data.drop(["id"],axis=1,inplace=True)
data= data.sample(frac=1)
x = naive_bayes(data,"num")
x.normalize(["age", "trestbps", "chol", "thalch", "oldpeak"])
x.train()
x.test()

# df = (data[["age"]] - data[["age"]].mean())/data[["age"]].std() 

# print(df.tail(5))
# print(df[["age"]].mean())
# print("Wjat",(53-data[["age"]].mean())/data[["age"]].std())
