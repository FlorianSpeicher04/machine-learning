import os
import numpy as np
import pandas as pd
from sklearn import set_config

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import MultiLabelBinarizer
import ast


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


set_config(transform_output="pandas") # dataframe supremacy

def prepDataset(dataset): #returns X_train, X_test, y_train, y_test
    dataset = pd.read_csv("./games_march2025_cleaned_2k.csv",sep=",")
    # desc, genres, tags
    column_transformer = ColumnTransformer([
            # merge all descriptions
            ('desc', FunctionTransformer(lambda X: X.fillna('').agg(' '.join, axis=1).to_frame(name="desc")),
                ['detailed_description', 'about_the_game', 'short_description']),
            ('pass', 'passthrough', ['genres']),#, 'tags'
        ],
        verbose_feature_names_out=False
    )
    dataset = column_transformer.fit_transform(dataset)



    #### SET MISSING VALUES
    print("SETMISS")
    # Setting missing numeric values to the mean
    dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
    # Setting missing text values to 'Unknown'
    dataset.fillna('', inplace=True)
    # Setting missing values in other columns to NaN
    dataset.dropna(inplace=True)

    ##### STRUCTURIZE GENRES to onehot
    #serialize array
    dataset['genres'] = dataset['genres'].map(lambda s: ast.literal_eval(s)) 
    #print(dataset['genres']) # in py but not yet onehotenc

    # MultiLabelBinarizer does onehotenc for arrays
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = mlb_genres.fit_transform(dataset.pop('genres'))
    #genres_count = len(mlb_genres.classes_) # for multi-label classifiction later

    genres_df = pd.DataFrame(genres_encoded, columns=mlb_genres.classes_)
    #print(genres_df)
    #dataset = pd.concat([dataset, genres_df], axis=1)
    #print(dataset)


    #### convert text to bag of words

    ## Count vs Tfidf vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dataset['desc']) # matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    #print(tfidf_df)


    ##### MODEL
    print("MODEL")


    X = tfidf_df
    y = genres_df
    # cleanup datapoints that dont have a target value (all target columns are 0)
    mask = y.sum(axis=1).map(lambda x: x > 0)
    #print((mask == False).sum()) #31 cases with all target columns 0
    X_clean = X[mask]
    y_clean = y[mask]

    # Split dataset
    return train_test_split(X_clean, y_clean, random_state=0)

def comparison(X_train, X_test, y_train, y_test, estimator, jobs: int = 1): #returns class_report
    multi_target_clf = MultiOutputClassifier(estimator, n_jobs=jobs) # LogisticRegression(max_iter=1337, random_state=0)

    # model training
    multi_target_clf.fit(X_train, y_train)

    # predict against test data
    y_pred = multi_target_clf.predict(X_test)
    return classification_report(y_test, y_pred, zero_division=0.0)

datasets = [
    'games_march2025_cleaned_2k.csv',
    'games_march2025_cleaned_10k.csv',
    'games_march2025_cleaned.csv'
]

estimators = {
    "LogisticRegression-i1000": LogisticRegression(max_iter=1000, random_state=0),
    "LogisticRegression-i10000": LogisticRegression(max_iter=10000, random_state=0),
    "LinearSVC-i5000": LinearSVC(max_iter=5000),
    "SVC-RBF-i10000": SVC(kernel="rbf", max_iter=10000),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=0),
    "RandomForestClassifier": RandomForestClassifier(random_state=0),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=0),
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "MLPClassifier-i10000": MLPClassifier(max_iter=10000, random_state=0),
}

for dataset in datasets:
    print("-" * 60)
    print("dataset -> " + dataset)
    print("-" * 60)
    print("mkdir")
    folder = dataset.split(".csv")[0]
    if not os.path.isdir(folder):
        os.mkdir(folder)
    X_train, X_test, y_train, y_test = prepDataset(dataset)
    for esti in estimators:
        compari = comparison(X_train, X_test, y_train, y_test, estimators[esti], 1) #TODO: change the job count if you can
        print("open")
        f = open(folder + "/" + esti +".txt", mode="w+", encoding="utf-8")
        f.write(compari)
        print("write")
        f.close()
        print("close")