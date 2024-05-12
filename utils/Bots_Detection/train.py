import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Load the dataset
raw_df = pd.read_csv("../../data/ecommerce_human_bot.csv")

# Extract features from 'created_at'
raw_df['created_at'] = pd.to_datetime(raw_df['created_at'])
raw_df['created_year'] = raw_df['created_at'].dt.year
raw_df['created_month'] = raw_df['created_at'].dt.month
raw_df['created_day'] = raw_df['created_at'].dt.day
raw_df['created_hour'] = raw_df['created_at'].dt.hour

# Binary classifications for bots and boolean values
raw_df['bot'] = raw_df['account_type'].apply(lambda x: 1 if x == 'bot' else 0)
raw_df['has_default_profile'] = raw_df['has_default_profile'].astype(int)
raw_df['has_default_profile_img'] = raw_df['has_default_profile_img'].astype(
    int)
raw_df['is_geo_enabled'] = raw_df['is_geo_enabled'].astype(int)

# Drop individual categorical columns and other non-numeric columns
raw_df.drop(columns=['user_id', 'user_lang', 'user_location',
                     'username', 'created_at', 'account_type'], inplace=True)


# Define feature set X and target variable y
X = raw_df.drop(['bot'], axis=1)
y = raw_df['bot']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=5)
lr = LogisticRegression(max_iter=1000)
gnb = GaussianNB()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
xgb = XGBClassifier()

# List of models
model_list = [knn, lr, tree, forest, xgb]

# Train and print accuracy for each model
for model in model_list:
    model.fit(X_train, y_train)
    accuracy = model.score(X_train, y_train)
    print(f"Model: {type(model).__name__}")
    print(f"Accuracy: {accuracy:.5f}")
    print()

# Save trained models
model_names = ['knn', 'lr', 'tree', 'forest', 'xgb']
for model, name in zip(model_list, model_names):
    with open(f'bots_detection_{name}_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

print("Training Done Successfully...")
