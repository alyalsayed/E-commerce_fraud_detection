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
raw_df = raw_df.iloc[[0]]
# Extract features from 'created_at'
raw_df['created_at'] = pd.to_datetime(raw_df['created_at'])
raw_df['created_year'] = raw_df['created_at'].dt.year
raw_df['created_month'] = raw_df['created_at'].dt.month
raw_df['created_day'] = raw_df['created_at'].dt.day
raw_df['created_hour'] = raw_df['created_at'].dt.hour


# Drop individual categorical columns and other non-numeric columns
raw_df.drop(columns=['user_id', 'user_lang', 'user_location',
                     'username', 'created_at', 'account_type'], inplace=True)


# Define a mapping for the labels
label_map = {0: 'human', 1: 'bot'}

# Predict using each model
for name, model in models.items():
    predictions = model.predict(raw_df)
    # Map predictions to labels
    predictions = [label_map[prediction] for prediction in predictions]
    print(f"Model: {name}")
    print("Predictions:", predictions[0])
    print()

print("Prediction Done Successfully...")
