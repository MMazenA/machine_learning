import os
from sklearn.model_selection import train_test_split
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score
import string
import matplotlib.pyplot as plt

def read_text_files(directory, label):
    texts = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='ISO-8859-1') as file:
                texts.append(file.read())
                labels.append(label)
    return texts, labels

def read_jsonl_file(file, label):
    data = []
    labels = []
    with open(file) as f:
        for line in f:
            json_data = json.loads(line)
            data.append(json_data)
            labels.append(label)
    return data, labels

def read_proccessed_data(file):
    if "sentence" in file:
        swap = "sentence"
    else:
        swap = "article"
    df=pd.read_csv(file,index_col=0)
    df = df.rename(columns={swap:"text","class":"label"})
    return df
    
def check_files_exist():
    current_directory = os.getcwd()  

    file1_path = os.path.join(current_directory, "test_data.csv")
    file2_path = os.path.join(current_directory, "train_data.csv")
    file3_path = os.path.join(current_directory, "val_data.csv")

    if all(os.path.exists(file_path) for file_path in [file1_path, file2_path, file3_path]):
        return True
    return False
    
def preproccess():
    if(check_files_exist()):
        return 
    
    human_text_data, human_labels = read_text_files('newspapers_data', label=0)
    ai_text_data, ai_labels = read_jsonl_file("175b_samples.jsonl", label=1)

    all_text_data = human_text_data + ai_text_data
    all_labels = human_labels + ai_labels
    df = pd.DataFrame({'text': all_text_data, 'label': all_labels})

    df_premade1 = read_proccessed_data("article_level_data.csv")
    df_premade2 = read_proccessed_data("sentence_level_data.csv")

    df = pd.concat([df_premade1, df_premade2,df], axis=0)



    df = df.sample(frac=1, random_state=11).reset_index(drop=True)
    df['label'].astype(str).str.replace('[{}]'.format(string.punctuation), '', regex=True)
    df.to_csv('human_gpt_classification.csv', index=False)

    train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=11)
    val_df, test_df = train_test_split(test_val_df, test_size=2/3, random_state=11)

    # save to file to debug
    train_df.to_csv('train_data.csv', index=False)
    val_df.to_csv('val_data.csv', index=False)
    test_df.to_csv('test_data.csv', index=False)

    print("Training set size:", len(train_df))
    print("Validation set size:", len(val_df))
    print("Testing set size:", len(test_df))

def main():
    preproccess()  
    train_df = pd.read_csv('train_data.csv')
    vali_df = pd.read_csv('val_data.csv')
    test_df = pd.read_csv('val_data.csv')

    # tokenize the data
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df['text'])
    X_val = vectorizer.transform(test_df['text']) #test values
    y_train = train_df['label']
    y_val = test_df['label']#test values

    # svm
    svm_model = SVC(kernel='linear', C=0.6)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_val)

    # random forest
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_val)

    # naive bayes
    nb_model = MultinomialNB(alpha=0.007)
    nb_model.fit(X_train, y_train)
    nb_predictions = nb_model.predict(X_val)

    # validation or testing results

    # f1-scores
    svm_f1_score = f1_score(y_val, svm_predictions, average='weighted')
    rf_f1_score = f1_score(y_val, rf_predictions, average='weighted')
    nb_f1_score = f1_score(y_val, nb_predictions, average='weighted')

    # matrices
    svm_conf_matrix = confusion_matrix(y_val, svm_predictions)
    rf_conf_matrix = confusion_matrix(y_val, rf_predictions)
    nb_conf_matrix = confusion_matrix(y_val, nb_predictions)

    total_instances = len(y_val)

    # stats
    print("Support Vector Machine:")
    print("Accuracy:", accuracy_score(y_val, svm_predictions))
    print("F1-Score:", svm_f1_score)
    print("Confusion Matrix:\n", svm_conf_matrix / total_instances )

    print("\nRandom Forest:")
    print("Accuracy:", accuracy_score(y_val, rf_predictions))
    print("F1-Score:", rf_f1_score)
    print("Confusion Matrix:\n", rf_conf_matrix / total_instances)

    print("\nMultinomial Naive Bayes:")
    print("Accuracy:", accuracy_score(y_val, nb_predictions))
    print("F1-Score:", nb_f1_score)
    print("Confusion Matrix:\n", nb_conf_matrix / total_instances)

    
    models = ['Support Vector Machine', 'Random Forest', 'Multinomial Naive Bayes']
    accuracies = [
        accuracy_score(y_val, svm_predictions),
        accuracy_score(y_val, rf_predictions),
        accuracy_score(y_val, nb_predictions)
    ]

    fig, ax = plt.subplots()
    ax.bar(models, accuracies, color=['blue', 'green', 'red'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracies on Test Data')
    plt.show()

    fig, ax = plt.subplots()
    ax.bar(models, accuracies, color=['blue', 'green', 'red'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracies on Validation Data')
    ax.set_ylim(0.7, 1.0)

    plt.show()





if __name__ == "__main__":
    main()
