from sklearn.model_selection import train_test_split, KFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pickle


raw_df = pd.read_csv("../../data/twitter_human_bots_dataset.csv", index_col=0)
print("Done uploading twitter_human_bots_dataset...")

# Binary classifications for bots and boolean values
raw_df['bot'] = raw_df['account_type'].apply(lambda x: 1 if x == 'bot' else 0)
raw_df['default_profile'] = raw_df['default_profile'].astype(int)
raw_df['default_profile'] = raw_df['default_profile'].astype(int)
raw_df['default_profile_image'] = raw_df['default_profile_image'].astype(int)
raw_df['geo_enabled'] = raw_df['geo_enabled'].astype(int)
raw_df['verified'] = raw_df['verified'].astype(int)


# datetime conversion
raw_df['created_at'] = pd.to_datetime(raw_df['created_at'])
# hour created
raw_df['hour_created'] = pd.to_datetime(raw_df['created_at']).dt.hour


df = raw_df[['bot', 'screen_name', 'created_at', 'hour_created', 'verified', 'location', 'geo_enabled', 'lang',
             'default_profile', 'default_profile_image', 'favourites_count', 'followers_count', 'friends_count',
             'statuses_count', 'average_tweets_per_day', 'account_age_days']]


del raw_df


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
    np.log(1 + (df['friends_count'] / df['account_age_days'])), 3)


print("Information about Dataset after feature engineering...\n")
df.info()


# Features selected for modeling
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

# Define feature set X and target variable y
X = df[features]
y = df['bot']


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("Done splitting data ...")

# Check the shape of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Initialize models
knn = KNeighborsClassifier(n_neighbors=10)
lr = LogisticRegression()
gnb = GaussianNB()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
xgb = XGBClassifier()

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the feature set
X_train_scaled = scaler.fit_transform(X)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
# List of models
model_list = [knn, lr, gnb, tree, forest, xgb]

# Initialize KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=33)

# Evaluation function


def multi_model_eval(model_list, X, y, kf):
    for model in model_list:
        accuracy_scores = cross_val_score(
            model, X, y, cv=kf, scoring='accuracy')
        precision_scores = cross_val_score(
            model, X, y, cv=kf, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=kf, scoring='recall')
        f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1')
        roc_auc_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')

        print(f"Model: {type(model).__name__}")
        print("-" * 30)
        print(
            f"Accuracy:  {accuracy_scores.mean():.5f} +- {accuracy_scores.std():.6f}")
        print(
            f"Precision: {precision_scores.mean():.5f} +- {precision_scores.std():.6f}")
        print(
            f"Recall:    {recall_scores.mean():.5f} +- {recall_scores.std():.6f}")
        print(f"F1 Score:  {f1_scores.mean():.5f} +- {f1_scores.std():.6f}")
        print(
            f"ROC AUC:   {roc_auc_scores.mean():.5f} +- {roc_auc_scores.std():.6f}")
        print()


# Train models
for model in model_list:
    model.fit(X_train_scaled, y)

# Save trained models
model_names = ['knn', 'lr', 'gnb', 'tree', 'forest', 'xgb']
for model, name in zip(model_list, model_names):
    with open(f'bots_detection_{name}_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

# Load models
loaded_models = []
for name in model_names:
    with open(f'bots_detection_{name}_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
        loaded_models.append(loaded_model)

# Print evaluation metrics
multi_model_eval(loaded_models, X_train_scaled, y, kf)

print("Training Done Successfully...")
