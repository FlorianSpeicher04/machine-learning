import numpy as np
import pandas as pd


#### INITIALIZE

# load data
# appid,name,release_date,required_age,price,dlc_count,detailed_description,about_the_game,short_description,reviews,header_image,website,support_url,support_email,windows,mac,linux,metacritic_score,metacritic_url,achievements,recommendations,notes,supported_languages,full_audio_languages,packages,developers,publishers,categories,genres,screenshots,movies,user_score,score_rank,positive,negative,estimated_owners,average_playtime_forever,average_playtime_2weeks,median_playtime_forever,median_playtime_2weeks,discount,peak_ccu,tags,pct_pos_total,num_reviews_total,pct_pos_recent,num_reviews_recent
dataset = pd.read_csv("./games_march2025_cleaned_10k.csv",sep=",")
print(dataset.head())




#### DROP UNIQUES

# appid,name,release_date,required_age,price,dlc_count,detailed_description,about_the_game,short_description,reviews,header_image,website,support_url,support_email,windows,mac,linux,metacritic_score,metacritic_url,achievements,recommendations,notes,supported_languages,full_audio_languages,packages,developers,publishers,categories,genres,screenshots,movies,user_score,score_rank,positive,negative,estimated_owners,average_playtime_forever,average_playtime_2weeks,median_playtime_forever,median_playtime_2weeks,discount,peak_ccu,tags,pct_pos_total,num_reviews_total,pct_pos_recent,num_reviews_recent
dataset.drop(['appid', 'name', 'release_date', 'reviews', 'header_image', 'website', 'support_url', 'support_email',
              'metacritic_url', 'notes', 'developers', 'publishers', 'screenshots', 'movies', 'estimated_owners'],
              axis=1, inplace=True)
print(dataset.head())






#### SET MISSING VALUES


# Setting missing numeric values to the mean
dataset.fillna(dataset.mean(numeric_only=True), inplace=True)
# Setting missing text values to 'Unknown'
dataset.fillna('Unknown', inplace=True)
# Setting missing values in other columns to NaN
dataset.dropna(inplace=True)





#### STRUCTURIZE AND STANDARDIZE

from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder

# appid,name,release_date,required_age,price,dlc_count,detailed_description,about_the_game,short_description,header_image,website,support_url,support_email,windows,mac,linux,metacritic_score,metacritic_url,achievements,recommendations,supported_languages,full_audio_languages,packages,developers,publishers,categories,genres,screenshots,movies,user_score,score_rank,positive,negative,estimated_owners,average_playtime_forever,average_playtime_2weeks,median_playtime_forever,median_playtime_2weeks,discount,peak_ccu,tags,pct_pos_total,num_reviews_total,pct_pos_recent,num_reviews_recent
column_transformer = make_column_transformer(
    (TfidfVectorizer(stop_words='english'), ['detailed_description']),
    (TfidfVectorizer(stop_words='english'), ['about_the_game']),
    (TfidfVectorizer(stop_words='english'), ['short_description']),
    (OneHotEncoder(), ['windows', 'mac', 'linux']),
    (StandardScaler(), ['price']),
    (FunctionTransformer(lambda x: x/100.0), ['metacritic_score']),
    (StandardScaler(), ['achievements']),
    (StandardScaler(), ['recommendations']),
    #TODO: custom onehot encoder for these:
    ('passthrough', ['supported_languages','full_audio_languages','categories','genres','tags']),
    ('passthrough', ['required_age', 'dlc_count','user_score','score_rank','positive','negative','average_playtime_forever','average_playtime_2weeks','median_playtime_forever','median_playtime_2weeks','discount','peak_ccu','pct_pos_total','num_reviews_total','pct_pos_recent','num_reviews_recent'])
)

dataset = column_transformer.fit_transform(dataset)
print(dataset.head())






#####


from sklearn.model_selection import train_test_split

# Annahme: 'genres' ist das Ziel/Label
X = dataset.drop('genres', axis=1)
y = dataset['genres']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Trainingsdaten: {X_train.shape}, Testdaten: {X_test.shape}")