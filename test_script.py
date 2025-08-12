

#### INITIALIZE

import numpy as np
import pandas as pd
from sklearn import set_config
set_config(transform_output="pandas") # dataframe supremacy

# load data
# appid,name,release_date,required_age,price,dlc_count,detailed_description,about_the_game,short_description,reviews,header_image,website,support_url,support_email,windows,mac,linux,metacritic_score,metacritic_url,achievements,recommendations,notes,supported_languages,full_audio_languages,packages,developers,publishers,categories,genres,screenshots,movies,user_score,score_rank,positive,negative,estimated_owners,average_playtime_forever,average_playtime_2weeks,median_playtime_forever,median_playtime_2weeks,discount,peak_ccu,tags,pct_pos_total,num_reviews_total,pct_pos_recent,num_reviews_recent
dataset = pd.read_csv("./games_march2025_cleaned_2k.csv",sep=",")
print(dataset.head())




#### DROP UNIQUES
print("DROP")

#TODO: wird eh unten beim transformer deleted

# appid,name,release_date,required_age,price,dlc_count,detailed_description,about_the_game,short_description,reviews,header_image,website,support_url,support_email,windows,mac,linux,metacritic_score,metacritic_url,achievements,recommendations,notes,supported_languages,full_audio_languages,packages,developers,publishers,categories,genres,screenshots,movies,user_score,score_rank,positive,negative,estimated_owners,average_playtime_forever,average_playtime_2weeks,median_playtime_forever,median_playtime_2weeks,discount,peak_ccu,tags,pct_pos_total,num_reviews_total,pct_pos_recent,num_reviews_recent
#dataset.drop(['appid', 'name', 'release_date', 'reviews', 'header_image', 'website', 'support_url', 'support_email',
#              'metacritic_url', 'notes', 'developers', 'publishers', 'screenshots', 'movies', 'estimated_owners'],
#              axis=1, inplace=True)
#print(dataset.head())

#### STRUCTURIZE AND STANDARDIZE
print("STRUCTURE")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer


# desc, genres, tags
column_transformer = ColumnTransformer([
        # merge all descriptions
        ('desc', FunctionTransformer(lambda X: X.fillna('').agg(' '.join, axis=1).to_frame(name="desc")),
            ['detailed_description', 'about_the_game', 'short_description']),
        # genre -> actual genre, but very coarse
        # tags -> user defined tags; title num list
        #TODO: decide whether we drop tags
        ('pass', 'passthrough', ['genres']),#, 'tags'
    ],
    verbose_feature_names_out=False
)
dataset = column_transformer.fit_transform(dataset)
print(dataset)



#### SET MISSING VALUES
print("SETMISS")


# Setting missing numeric values to the mean
dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
# Setting missing text values to 'Unknown'
dataset.fillna('', inplace=True)
# Setting missing values in other columns to NaN
dataset.dropna(inplace=True)




##### STRUCTURIZE GENRES to onehot
import ast
#serialize array
dataset['genres'] = dataset['genres'].map(lambda s: ast.literal_eval(s)) 
print(dataset['genres']) # in py but not yet onehotenc

# MultiLabelBinarizer does onehotenc for arrays
mlb_genres = MultiLabelBinarizer()
genres_encoded = mlb_genres.fit_transform(dataset.pop('genres'))
genres_count = len(mlb_genres.classes_) # for multi-label classifiction later

genres_df = pd.DataFrame(genres_encoded, columns=mlb_genres.classes_)
print(genres_df)
#dataset = pd.concat([dataset, genres_df], axis=1)
#print(dataset)


#### convert text to bag of words

## Count vs Tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset['desc']) # matrix
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print(tfidf_df)


##### MODEL
print("MODEL")

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


X = tfidf_df
y = genres_df


# cleanup datapoints that dont have a target value (all target columns are 0)
mask = y.sum(axis=1).map(lambda x: x > 0)
#print((mask == False).sum()) #31 cases with all target columns 0

X_clean = X[mask]
y_clean = y[mask]


print(X_clean)
print(y_clean)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, random_state=0)


print(X_train)
print(y_train)

# we want to have multiple possible outputs (multi-label-classficiation) -> multioutputclassifier
# logi regression is our base system
# n_jobs=1 since there seems to be some multithreading join issue in sklearn
multi_target_clf = MultiOutputClassifier(LogisticRegression(max_iter=1337, random_state=0), n_jobs=1)

# model training
multi_target_clf.fit(X_train, y_train)

# predict against test data
y_pred = multi_target_clf.predict(X_test)

# classify
print(classification_report(y_test, y_pred, zero_division=0.0))


#print(f"Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")
