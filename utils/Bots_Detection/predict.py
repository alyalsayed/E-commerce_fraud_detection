import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# Load the trained models
model_names = ['knn', 'lr', 'tree', 'forest', 'xgb']
models = {}
for name in model_names:
    with open(f'bots_detection_{name}_model.pkl', 'rb') as model_file:
        models[name] = pickle.load(model_file)

# Load the dataset
raw_df = pd.read_csv("../../data/ecommerce_human_bot.csv")
# Selecting only the first row for prediction
raw_df = raw_df.iloc[[4]]

# Preprocess data


def preprocess_data(df):
    # Extract features from 'created_at'
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['created_year'] = df['created_at'].dt.year
    df['created_month'] = df['created_at'].dt.month
    df['created_day'] = df['created_at'].dt.day
    df['created_hour'] = df['created_at'].dt.hour

    # Binary classifications for bots and boolean values
    df['bot'] = df['account_type'].apply(lambda x: 1 if x == 'bot' else 0)
    df['has_default_profile'] = df['has_default_profile'].astype(int)
    df['has_default_profile_img'] = df['has_default_profile_img'].astype(int)
    df['is_geo_enabled'] = df['is_geo_enabled'].astype(int)

    # Drop individual categorical columns and other non-numeric columns
    df.drop(columns=['user_id', 'user_lang', 'user_location',
                     'username', 'created_at', 'account_type'], inplace=True)
    df = df[['has_default_profile', 'has_default_profile_img', 'prod_fav_count',
             'followers_count', 'friends_count', 'is_geo_enabled', 'purchase_count',
             'membership / subscription', 'avg_purchases_per_day', 'account_age',
             'created_year', 'created_month', 'created_day', 'created_hour', 'bot']]


preprocess_data(raw_df)
X_pred = raw_df.drop(['bot'], axis=1)
# Define a mapping for the labels
label_map = {0: 'human', 1: 'bot'}

# Predict using each model
for name, model in models.items():
    predictions = model.predict(X_pred)
    # Map predictions to labels
    predictions = [label_map[prediction] for prediction in predictions]
    print(f"Model: {name}")
    print("Predictions:", predictions[0])
    print()

print("Prediction Done Successfully...")
