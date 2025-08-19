import os
import numpy as np
import pandas as pd
from sklearn import set_config
import gc

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier

set_config(transform_output="pandas") # dataframe supremacy

jobs = 12
max_iter = 3000
min_entries = 5

def prepDataset(dataset): #returns X_train, X_test, y_train, y_test
    dataset = pd.read_csv(dataset,sep=",")
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

    # remove genres that have less than min_entries entries -> probability of broken split to big
    mask = (y == 1).sum() >= min_entries
    print(y.shape)
    y_prep = y.loc[:, mask]
    print(y_prep.shape)
    del mask
    del y

    # cleanup datapoints that dont have a target value (all target columns are 0)
    mask = y_prep.sum(axis=1).map(lambda x: x > 0)
    #print((mask == False).sum()) #31 cases with all target columns 0
    X_clean = X[mask]
    y_clean = y_prep[mask]

    # clean ram edition
    del dataset
    del column_transformer #-
    del mlb_genres
    del genres_encoded
    del genres_df #-
    del tfidf_df
    del vectorizer
    del tfidf_matrix #-
    del X
    del y_prep
    del mask
    gc.collect()

    # Split dataset
    return train_test_split(X_clean, y_clean, random_state=0)
def comparison(X_train, X_test, y_train, y_test, estimator,): #returns class_report
    multi_target_clf = MultiOutputClassifier(estimator, n_jobs=jobs) # LogisticRegression(max_iter=1337, random_state=0)
    # model training
    multi_target_clf.fit(X_train, y_train)
    # predict against test data
    y_pred = multi_target_clf.predict(X_test)
    return classification_report(y_test, y_pred, zero_division=0.0)
datasets = [
    #'games_march2025_cleaned_2k.csv',
    'games_march2025_cleaned_10k.csv',
    #'games_march2025_cleaned.csv'
]
estimators = {
    #"RidgeClassifier": RidgeClassifier(random_state=0, max_iter=max_iter),
    #"PassiveAggressiveClassifier": PassiveAggressiveClassifier(random_state=0, max_iter=max_iter),
    "Perceptron": Perceptron(random_state=0, max_iter=max_iter),
    "SGDClassifier": SGDClassifier(random_state=0, max_iter=max_iter),
    "NearestCentroid": NearestCentroid(),
    "LinearSVC": LinearSVC(random_state=0, max_iter=max_iter),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=0),
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=0, max_iter=max_iter),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "MLPClassifier": MLPClassifier(random_state=0, max_iter=int(max_iter/20), early_stopping=True),
}

#"VotingClassifier": VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier())]),
#"StackingClassifier": StackingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier())]),
for dataset in datasets:
    print("-" * 60)
    print("dataset -> " + dataset)
    print("mkdir")
    folder = dataset.split(".csv")[0]
    if not os.path.isdir(folder):
        os.mkdir(folder)
    X_train, X_test, y_train, y_test = prepDataset(dataset)
    for esti in estimators:
        print("model: " + esti)
        compari = comparison(X_train, X_test, y_train, y_test, estimators[esti])
        print("open")
        f = open(folder + "/" + esti +".txt", mode="w+", encoding="utf-8")
        f.write(compari)
        print("write")
        f.close()
        print("close")