import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Load the trained models
model_names = ['knn', 'lr', 'tree', 'forest', 'xgb']
models = {}
for name in model_names:
    with open(f'bots_detection_{name}_model.pkl', 'rb') as model_file:
        models[name] = pickle.load(model_file)

# Load the dataset
raw_df = pd.read_csv("../../data/ecommerce_human_bot.csv")
raw_df = raw_df.iloc[[21]]
# Extract features from 'created_at'
raw_df['created_at'] = pd.to_datetime(raw_df['created_at'])
raw_df['created_year'] = raw_df['created_at'].dt.year
raw_df['created_month'] = raw_df['created_at'].dt.month
raw_df['created_day'] = raw_df['created_at'].dt.day
raw_df['created_hour'] = raw_df['created_at'].dt.hour

# Concatenate categorical columns into 'user_info'
raw_df['user_info'] = raw_df['user_lang'] + "_" + raw_df['user_location']

# Drop individual categorical columns and other non-numeric columns
raw_df.drop(columns=['user_id', 'user_lang', 'user_location',
                     'username', 'created_at', 'account_type'], inplace=True)

# Apply TF-IDF vectorizer only on 'user_info' column
X_tfidf = tfidf_vectorizer.transform(raw_df['user_info'])

# Scale numeric columns
numeric_cols = raw_df.select_dtypes(include=['int64', 'float64']).columns
X_numeric_scaled = scaler.transform(raw_df[numeric_cols])

# Merge TF-IDF features with numeric columns
X_final = hstack([X_tfidf, X_numeric_scaled])

# Define a mapping for the labels
label_map = {0: 'human', 1: 'bot'}

# Predict using each model
for name, model in models.items():
    predictions = model.predict(X_final)
    # Map predictions to labels
    predictions = [label_map[prediction] for prediction in predictions]
    print(f"Model: {name}")
    print("Predictions:", predictions[0])
    print()

print("Prediction Done Successfully...")
