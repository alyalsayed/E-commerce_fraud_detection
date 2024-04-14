import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load the dataset
raw_df = pd.read_csv("../../data/ecommerce_human_bot.csv")

# Extract features from 'created_at'
raw_df['created_at'] = pd.to_datetime(raw_df['created_at'])
raw_df['created_year'] = raw_df['created_at'].dt.year
raw_df['created_month'] = raw_df['created_at'].dt.month
raw_df['created_day'] = raw_df['created_at'].dt.day
raw_df['created_hour'] = raw_df['created_at'].dt.hour

# Concatenate categorical columns into 'user_info'
raw_df['user_info'] = raw_df['user_lang'] + "_" + raw_df['user_location']

# Binary classifications for bots and boolean values
raw_df['bot'] = raw_df['account_type'].apply(lambda x: 1 if x == 'bot' else 0)
raw_df['has_default_profile'] = raw_df['has_default_profile'].astype(int)
raw_df['has_default_profile_img'] = raw_df['has_default_profile_img'].astype(
    int)
raw_df['is_geo_enabled'] = raw_df['is_geo_enabled'].astype(int)

# Drop individual categorical columns and other non-numeric columns
raw_df.drop(columns=['user_lang', 'user_location',
            'username', 'created_at', 'account_type'], inplace=True)

# Define feature set X and target variable y
X = raw_df.drop(['bot'], axis=1)
y = raw_df['bot']

# Load TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    min_df=1, stop_words='english', lowercase=True)

# Fit TF-IDF vectorizer on 'user_info' column
user_info_tfidf = tfidf_vectorizer.fit_transform(X['user_info'])

# Load trained models
model_names = ['knn', 'lr', 'tree', 'forest', 'xgb']
loaded_models = []
for name in model_names:
    with open(f'bots_detection_{name}_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
        loaded_models.append(loaded_model)

# Assume we have a single test record (first record from the dataset)
test_record = X.iloc[[0]]

# Apply TF-IDF vectorizer on 'user_info' column of the test record
test_record_features = tfidf_vectorizer.transform(test_record['user_info'])

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Scale numeric columns of the test record
numeric_cols = test_record.select_dtypes(include=['int64', 'float64']).columns
test_record[numeric_cols] = scaler.transform(test_record[numeric_cols])

# Predict using loaded models
print("Predictions:")
for model, name in zip(loaded_models, model_names):
    prediction = model.predict(test_record_features)
    print(
        f"Prediction using {name} model: {'bot' if prediction[0] == 1 else 'human'}")
