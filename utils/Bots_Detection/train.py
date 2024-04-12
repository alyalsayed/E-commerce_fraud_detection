


# #  Bots detection

# ## DataFrame Description:
# 
# ***This DataFrame contains data related to Twitter accounts and their attributes, along with a column indicating whether each account is detected as a bot.****
# 
# * created_at: Timestamp indicating when the account was created.
# * default_profile: Boolean indicating if the account has the default profile settings.
# * default_profile_image: Boolean indicating if the account has the default profile image.
# * description: Description of the account (bio).
# * favourites_count: Number of tweets the account has favorited.
# * followers_count: Number of followers of the account.
# * friends_count: Number of accounts the account is following.
# * geo_enabled: Boolean indicating if the account has enabled geolocation.
# * id: Unique identifier for the account.
# * lang: Language of the account.
# * location: Location specified by the account.
# * profile_background_image_url: URL of the background image for the account's profile.
# * profile_image_url: URL of the profile image for the account.
# * screen_name: Twitter handle of the account.
# * statuses_count: Number of tweets made by the account.
# * verified: Boolean indicating if the account is verified by Twitter.
# * average_tweets_per_day: Average number of tweets per day made by the account.
# * account_age_days: Age of the account in days.
# * account_type: Type of account.

# In[135]:


# Basics
import pandas as pd
import numpy as np
import pickle

# Visuals
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[136]:


raw_df = pd.read_csv("/kaggle/input/twitter-bots/twitter_human_bots_dataset.csv", index_col=0)
raw_df.head()


# In[137]:


raw_df.info()


# In[138]:


# Binary classifications for bots and boolean values
raw_df['bot'] = raw_df['account_type'].apply(lambda x: 1 if x == 'bot' else 0)
raw_df['default_profile'] = raw_df['default_profile'].astype(int)
raw_df['default_profile'] = raw_df['default_profile'].astype(int)
raw_df['default_profile_image'] = raw_df['default_profile_image'].astype(int)
raw_df['geo_enabled'] = raw_df['geo_enabled'].astype(int)
raw_df['verified'] = raw_df['verified'].astype(int)


# In[139]:


# datetime conversion
raw_df['created_at'] = pd.to_datetime(raw_df['created_at'])
# hour created
raw_df['hour_created'] = pd.to_datetime(raw_df['created_at']).dt.hour


# In[140]:


raw_df.info()


# In[141]:


df = raw_df[['bot', 'screen_name', 'created_at', 'hour_created', 'verified', 'location', 'geo_enabled', 'lang', 
             'default_profile', 'default_profile_image', 'favourites_count', 'followers_count', 'friends_count', 
             'statuses_count', 'average_tweets_per_day', 'account_age_days']]


# In[142]:


del raw_df


# In[143]:


# Interesting features to look at: 
df['avg_daily_followers'] = np.round((df['followers_count'] / df['account_age_days']), 3)
df['avg_daily_friends'] = np.round((df['followers_count'] / df['account_age_days']), 3)
df['avg_daily_favorites'] = np.round((df['followers_count'] / df['account_age_days']), 3)

# Log transformations for highly skewed data
df['friends_log'] = np.round(np.log(1 + df['friends_count']), 3)
df['followers_log'] = np.round(np.log(1 + df['followers_count']), 3)
df['favs_log'] = np.round(np.log(1 + df['favourites_count']), 3)
df['avg_daily_tweets_log'] = np.round(np.log(1+ df['average_tweets_per_day']), 3)

# Possible interactive features
df['network'] = np.round(df['friends_log'] * df['followers_log'], 3)
df['tweet_to_followers'] = np.round(np.log( 1 + df['statuses_count']) * np.log(1+ df['followers_count']), 3)

# Log-transformed daily acquisition metrics for dist. plots
df['follower_acq_rate'] = np.round(np.log(1 + (df['followers_count'] / df['account_age_days'])), 3)
df['friends_acq_rate'] = np.round(np.log(1 + (df['friends_count'] / df['account_age_days'])), 3)
df['favs_rate'] = np.round(np.log(1 + (df['friends_count'] / df['account_age_days'])), 3)


# In[144]:


df.head()


# In[145]:


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


# In[146]:


from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[147]:


# Check the shape of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[148]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Initialize models
knn = KNeighborsClassifier(n_neighbors=10)
lr = LogisticRegression()

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the feature set
X_train_scaled = scaler.fit_transform(X)

# List of models
model_list = [knn, lr]

# Initialize KFold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=33)


# In[149]:


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def multi_model_eval(model_list, X, y, kf):
    """
    Evaluate multiple models using cross-validation.
    
    Parameters:
        model_list (list): List of classifier models to evaluate.
        X (array-like): Feature set.
        y (array-like): Target variable.
        kf (KFold): Cross-validation strategy.
    
    Returns:
        None (Prints evaluation results)
    """
    for model in model_list:
        accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=kf, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=kf, scoring='recall')
        f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1')
        roc_auc_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
        
        print(f"Model: {type(model).__name__}")
        print("-" * 30)
        print(f"Accuracy:  {accuracy_scores.mean():.5f} +- {accuracy_scores.std():.6f}")
        print(f"Precision: {precision_scores.mean():.5f} +- {precision_scores.std():.6f}")
        print(f"Recall:    {recall_scores.mean():.5f} +- {recall_scores.std():.6f}")
        print(f"F1 Score:  {f1_scores.mean():.5f} +- {f1_scores.std():.6f}")
        print(f"ROC AUC:   {roc_auc_scores.mean():.5f} +- {roc_auc_scores.std():.6f}")
        print()


# In[150]:


multi_model_eval(model_list, X_train_scaled, y, kf)


# In[151]:


knn.fit(X_train_scaled, y)
lr.fit(X_train_scaled, y)

# Train and save KNeighborsClassifier
with open('bots_detection_knn_model.pkl', 'wb') as knn_file:
    pickle.dump(knn, knn_file)

# Train and save LogisticRegression
with open('bots_detection_lr_model.pkl', 'wb') as lr_file:
    pickle.dump(lr, lr_file)


# In[152]:


# Load KNeighborsClassifier model
with open('bots_detection_knn_model.pkl', 'rb') as knn_file:
    knn_model = pickle.load(knn_file)

# Load LogisticRegression model
with open('bots_detection_lr_model.pkl', 'rb') as lr_file:
    lr_model = pickle.load(lr_file)


# In[153]:


import pickle
from sklearn.metrics import accuracy_score

# Scale the test features
X_test_scaled = scaler.transform(X_test)

# Predict using KNeighborsClassifier model
knn_predictions = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print("Accuracy of KNeighborsClassifier:", knn_accuracy)

# Predict using LogisticRegression model
lr_predictions = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print("Accuracy of LogisticRegression:", lr_accuracy)


# In[154]:


from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold

# Initialize models
gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()
tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
xgb = XGBClassifier()

# List of models
model_list = [gnb, bnb, mnb, tree, forest, xgb]

# Initialize KFold cross-validator
kf = KFold(n_splits=3, shuffle=True, random_state=33)


# In[155]:


multi_model_eval(model_list, X, y, kf)


# # Content Moderation

# ## * url: URL of the content.
# ## * title: Title of the content.
# ## * asin: ASIN (Amazon Standard Identification Number).
# ## * price: Price of the product.
# ## * brand: Brand of the product.
# ## * product_details: Details of the product.
# ## * breadcrumbs: Breadcrumbs for navigation.
# ## * features: Features of the product.
# ## * location: Location of the content.
# ## * has_company_logo: Indicates whether the content has a company logo.
# ## * has_questions: Indicates whether the content has questions.
# ## * industry: Industry of the content.
# ## * fraudulent: Indicates whether the content is fraudulent.

# In[156]:


import re
import string
import numpy as np
import pandas as pd
import random
import missingno
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from wordcloud import WordCloud

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


# In[157]:


data = pd.read_csv('/kaggle/input/content-moderation/content_moderation_dataset.csv', index_col=0)
data.head()


# In[158]:


data.features[0]


# In[159]:


import ast

# Function to preprocess the string representation of dictionaries
def preprocess_features_string(features_str):
    # Replace single quotes with double quotes
    features_str = features_str.replace("'", '"')
    return features_str

# Function to extract shoe materials from the preprocessed 'features' column
def extract_shoe_materials(features_str):
    # Preprocess the features string
    features_str = preprocess_features_string(features_str)
    
    try:
        # Parse the features string to a Python list of dictionaries
        features_list = ast.literal_eval(features_str)
    except (SyntaxError, ValueError):
        # Return an empty list if parsing fails
        return ''
    
    # List to store extracted materials
    materials = []
    
    # Iterate over each feature dictionary
    for feature in features_list:
        # Extract the value corresponding to 'Outer Material', 'Inner Material', 'Sole', and 'Closure' keys
        if 'Outer Material' in feature:
            materials.append(feature['Outer Material'])
        if 'Inner Material' in feature:
            materials.append(feature['Inner Material'])
        if 'Sole' in feature:
            materials.append(feature['Sole'])
        if 'Closure' in feature:
            materials.append(feature['Closure'])
    
    # Join the extracted materials with commas
    return ','.join(materials)

# Apply the function to clean the 'features' column
data['cleaned_features'] = data['features'].apply(extract_shoe_materials)


# In[160]:


print(data.columns)
data.describe()


# In[161]:


data.info()


# In[162]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'data' contains your DataFrame

# Count occurrences of each category in the 'fraudulent' column
fraudulent_counts = data['fraudulent'].value_counts()

# Plot the counts with default colors
plt.bar(fraudulent_counts.index, fraudulent_counts.values, color=['C0', 'C1'])

# Add labels and title
plt.xlabel('Fraudulent')
plt.ylabel('Count')
plt.title('Distribution of Fraudulent vs Non-Fraudulent Content')

# Add x-axis labels
plt.xticks(fraudulent_counts.index, ['Non-Fraudulent', 'Fraudulent'])

# Show plot
plt.show()


# In[163]:


# Fill missing values with empty strings
data.fillna('', inplace=True)


# In[164]:


data['text'] = data['title'] + ' ' + data['location'] + ' ' + data['brand'] +   ' ' + data['breadcrumbs'] +   ' '+ data['cleaned_features'] +' '+ data['industry']


# In[165]:


# Drop the specified columns from the DataFrame
columns_to_drop = ['title', 'location', 'brand', 'breadcrumbs', 'cleaned_features','features', 'industry','asin','has_company_logo','has_questions']
data.drop(columns=columns_to_drop, inplace=True)


# In[166]:


data.head()


# In[167]:


fraud_text = data[data.fraudulent==1].text
actual_text = data[data.fraudulent==0].text


# In[168]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16, 14))
wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800, stopwords=STOPWORDS).generate(str(" ".join(fraud_text)))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[169]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16, 14))
wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800, stopwords=STOPWORDS).generate(str(" ".join(actual_text)))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[170]:


import spacy
from spacy.lang.en import English
import string

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")
parser = English()

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Custom analyzer function to preprocess text
def spacy_tokenizer(sentence):
    # Parse the sentence using SpaCy
    doc = nlp(sentence)
    
    # Lemmatize each token and convert to lowercase
    mytokens = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in doc]
    
    # Remove stop words and punctuations
    mytokens = [token for token in mytokens if token not in stop_words and token not in punctuations]
    
    # Return preprocessed list of tokens
    return mytokens


# In[171]:


# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


# In[172]:


# Create CountVectorizer with custom analyzer
bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1,1))


# In[173]:


# splitting our data in train and test
X_train, X_test, y_train, y_test = train_test_split(data.text, data.fraudulent, test_size=0.2)


# In[174]:


def train_model(X_train, y_train, X_test, y_test, classifier):
    # Create pipeline using Bag of Words
    pipe = Pipeline([
        ("cleaner", predictors()),
        ('vectorizer', CountVectorizer(tokenizer=spacy_tokenizer)),
        ('classifier', classifier)
    ])

    # Fit the model
    pipe.fit(X_train, y_train)

    # Predict on the test set
    predicted = pipe.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, predicted)

    # Print accuracy
    print(f"Model Accuracy: {accuracy}")

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, predicted)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.title('Confusion Matrix')
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    # Return the trained model
    return pipe


# In[175]:


import pickle
# Define classifiers
classifiers = {
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "SVC": SVC(),
    "XGBClassifier": XGBClassifier()
}

# Train and save each model
for name, clf in classifiers.items():
    print(f"Training {name}...")
    trained_model = train_model(X_train, y_train, X_test, y_test, clf)
    # Save the trained model
    filename = f"{name}_content_moderation.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(trained_model, file)
    print(f"Model {name} saved as {filename}\n")

