from statistics import NormalDist
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
        self.continues_cols = continues_cols
        #map boolean features to strings
        mask = self.dataset.map(type) != bool
        replace = {True: 'TRUE', False: 'FALSE'}
        self.dataset = self.dataset.where(mask, self.dataset.replace(replace))
        
    def split_data(self, test_ratio=0.2):
        """Split the dataset into training and testing sets."""
        testing_size = int(self.dataset.shape[0] * test_ratio)
        self.testing_set = self.dataset.tail(testing_size)
        self.training_set = self.dataset.drop(self.testing_set.index)
        self.possible_outputs = self.dataset[self.predicting_feature].unique()

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
        """Produce every features mean and variance for continous features. Assumes variance is equal between classes."""
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

    def predict_single(self,row: pd.DataFrame):
        """Produces prediction for a single row."""
        sol = {}
        row_without_classifier = row.drop([self.predicting_feature],axis=1)        
        for key in self.possible_outputs:
            likelihood = np.log(self.occurance_prob[key] / sum(self.occurance_prob))

            for feature in row_without_classifier.columns:
                    if feature in self.continues_cols:
                        mstd = self.probabilities[key][feature]
                        mean = mstd[0]
                        std = mstd[1]
                        z_score = (row[feature].values - mean)/std
                        if z_score > 0:
                            z_score = -z_score
                        cdf = NormalDist().cdf(z_score) * 2                        
                        likelihood+=np.log(cdf)
                    else:
                        row_feature = row[feature]
                        if row_feature.values[0] not in self.probabilities[key][feature]:
                            #never before seen feature
                            continue
                        feature_occurance = self.probabilities[key][feature][row_feature.values].values
                        feature_total = sum(self.probabilities[key][feature])
                        probability = feature_occurance/feature_total
                        likelihood+= np.log(probability)
            sol[key] = likelihood
        
        maximum =["key",float("-inf")]
        for key in sol:
            if sol[key] > maximum[1]:
                maximum = [key,sol[key]]
        return maximum[0]
            
    def test(self, testing_set=None):
        """Run testing set to determine accuracy."""
        if testing_set:
            self.testing_set = testing_set
        row_count = self.testing_set.shape[0]
        wrong=0
        for i in range(row_count):
            prediction = self.predict_single(self.testing_set.iloc[[i]])
            if prediction==0 and self.testing_set.iloc[[i]][self.predicting_feature].values !=0:
                wrong+=1
        print(wrong,row_count)
        return [wrong,row_count]



def main():
    data = pd.read_csv("heart_disease_uci.csv")
    data.drop(["id"],axis=1,inplace=True)
    #randomize data
    data= data.sample(frac=1)
    nb = naive_bayes(data,"num")
    nb.normalize(["age", "trestbps", "chol", "thalch", "oldpeak"])
    nb.train()
    nb.test()

if __name__=="__main__":
    main()
