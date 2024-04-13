# forest model gives greate results
#
# Load the input data
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
# Load the dataset
raw_df = pd.read_csv("../../data/twitter_human_bots_dataset.csv", index_col=0)

# Extract the first two records as DataFrames
input_data_1 = raw_df.iloc[[0]]
input_data_2 = raw_df.iloc[[6710]]


# Load trained models
model_names = ['knn', 'lr', 'gnb', 'tree', 'forest', 'xgb']
loaded_models = []
for name in model_names:
    with open(f'bots_detection_{name}_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
        loaded_models.append(loaded_model)


def preprocess_and_feature_engineering(input_data):
    # Create DataFrame from input data
    df = pd.DataFrame(input_data)

    # Binary classifications for bots and boolean values
    df['bot'] = df['account_type'].apply(lambda x: 1 if x == 'bot' else 0)
    df['default_profile'] = df['default_profile'].astype(int)
    df['default_profile_image'] = df['default_profile_image'].astype(int)
    df['geo_enabled'] = df['geo_enabled'].astype(int)
    df['verified'] = df['verified'].astype(int)

    # datetime conversion
    df['created_at'] = pd.to_datetime(df['created_at'])
    # hour created
    df['hour_created'] = pd.to_datetime(df['created_at']).dt.hour

    # Interesting features to look at:
    df['avg_daily_followers'] = np.round(
        (df['followers_count'] / df['account_age_days']), 3)
    df['avg_daily_friends'] = np.round(
        (df['followers_count'] / df['account_age_days']), 3)
    df['avg_daily_favorites'] = np.round(
        (df['followers_count'] / df['account_age_days']), 3)

    # Log transformations for highly skewed data
    df['friends_log'] = np.round(np.log(1 + df['friends_count']), 3)
    df['followers_log'] = np.round(np.log(1 + df['followers_count']), 3)
    df['favs_log'] = np.round(np.log(1 + df['favourites_count']), 3)
    df['avg_daily_tweets_log'] = np.round(
        np.log(1 + df['average_tweets_per_day']), 3)

    # Possible interactive features
    df['network'] = np.round(df['friends_log'] * df['followers_log'], 3)
    df['tweet_to_followers'] = np.round(
        np.log(1 + df['statuses_count']) * np.log(1 + df['followers_count']), 3)

    # Log-transformed daily acquisition metrics for dist. plots
    df['follower_acq_rate'] = np.round(
        np.log(1 + (df['followers_count'] / df['account_age_days'])), 3)
    df['friends_acq_rate'] = np.round(
        np.log(1 + (df['friends_count'] / df['account_age_days'])), 3)
    df['favs_rate'] = np.round(
        np.log(1 + (df['favourites_count'] / df['account_age_days'])), 3)

    return df


# Preprocess and feature engineering
df_input = preprocess_and_feature_engineering(input_data_2)
print(df_input)
# Select features
features = [
    'verified',
    'geo_enabled',
    'default_profile',
    'default_profile_image',
    'favourites_count',
    'followers_count',
    'friends_count',
    'statuses_count',
    'average_tweets_per_day',
    'network',
    'tweet_to_followers',
    'follower_acq_rate',
    'friends_acq_rate',
    'favs_rate'
]

X_input = df_input[features]
print(X_input)
# Standardize the feature set

X_input_scaled = scaler.transform(X_input)

# Prediction
for model, name in zip(loaded_models, model_names):
    prediction = model.predict(X_input_scaled)
    print(
        f"Prediction using {name} model: {'bot' if prediction[0] == 1 else 'human'}")
