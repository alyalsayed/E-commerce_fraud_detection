
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


import pickle
import ast
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


data = pd.read_csv(
    '/kaggle/input/content-moderation/content_moderation_dataset.csv', index_col=0)
data.head()


# In[158]:


data.features[0]


# In[159]:


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


data['text'] = data['title'] + ' ' + data['location'] + ' ' + data['brand'] + \
    ' ' + data['breadcrumbs'] + ' ' + \
    data['cleaned_features'] + ' ' + data['industry']


# In[165]:


# Drop the specified columns from the DataFrame
columns_to_drop = ['title', 'location', 'brand', 'breadcrumbs', 'cleaned_features',
                   'features', 'industry', 'asin', 'has_company_logo', 'has_questions']
data.drop(columns=columns_to_drop, inplace=True)


# In[166]:


data.head()


# In[167]:


fraud_text = data[data.fraudulent == 1].text
actual_text = data[data.fraudulent == 0].text


# In[168]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16, 14))
wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800,
               stopwords=STOPWORDS).generate(str(" ".join(fraud_text)))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[169]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16, 14))
wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800,
               stopwords=STOPWORDS).generate(str(" ".join(actual_text)))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[170]:


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
    mytokens = [token.lemma_.lower().strip() if token.lemma_ !=
                "-PRON-" else token.lower_ for token in doc]

    # Remove stop words and punctuations
    mytokens = [
        token for token in mytokens if token not in stop_words and token not in punctuations]

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
bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))


# In[173]:


# splitting our data in train and test
X_train, X_test, y_train, y_test = train_test_split(
    data.text, data.fraudulent, test_size=0.2)


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
