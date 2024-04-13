import pickle
import ast
import re
import os
import string
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import random
import missingno
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer


import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('../../data/content_moderation_dataset.csv', index_col=0)

print("Done uploading content_moderation_dataset...")


def preprocess_features_string(features_str):
    # Replace single quotes with double quotes
    features_str = features_str.replace("'", '"')
    return features_str


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


data['cleaned_features'] = data['features'].apply(extract_shoe_materials)

data.fillna('', inplace=True)

data['text'] = data['title'] + ' ' + data['location'] + ' ' + data['brand'] + \
    ' ' + data['breadcrumbs'] + ' ' + \
    data['cleaned_features'] + ' ' + data['industry']

columns_to_drop = ['title', 'location', 'brand', 'breadcrumbs', 'cleaned_features',
                   'features', 'industry', 'asin', 'has_company_logo', 'has_questions']
data.drop(columns=columns_to_drop, inplace=True)

fraud_text = data[data.fraudulent == 1].text
actual_text = data[data.fraudulent == 0].text

STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16, 14))
wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800,
               stopwords=STOPWORDS).generate(str(" ".join(fraud_text)))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')

# Check if the directory exists, if not create it
if not os.path.exists("images"):
    os.makedirs("images")

# Save the image
plt.savefig("images/fraudulent_wordcloud.png", bbox_inches='tight')
plt.show()

STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize=(16, 14))
wc = WordCloud(min_font_size=3, max_words=3000, width=1600, height=800,
               stopwords=STOPWORDS).generate(str(" ".join(actual_text)))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')

# Save the image
plt.savefig("images/actual_wordcloud.png", bbox_inches='tight')
plt.show()

nlp = spacy.load("en_core_web_sm")
parser = English()
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS


def spacy_tokenizer(sentence):
    doc = nlp(sentence)
    mytokens = [token.lemma_.lower().strip() if token.lemma_ !=
                "-PRON-" else token.lower_ for token in doc]
    mytokens = [
        token for token in mytokens if token not in stop_words and token not in punctuations]
    return mytokens


class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def clean_text(text):
    return text.strip().lower()


bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    data.text, data.fraudulent, test_size=0.2)


def train_model(X_train, y_train, X_test, y_test, classifier):
    pipe = Pipeline([
        ("cleaner", predictors()),
        ('vectorizer', bow_vector),
        ('classifier', classifier)
    ])

    pipe.fit(X_train, y_train)
    predicted = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)

    print(f"Model Accuracy: {accuracy}")

    return pipe


classifiers = {
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "SVC": SVC(),
    "XGBClassifier": XGBClassifier()
}

for name, clf in classifiers.items():
    print(f"Training {name}...")
    trained_model = train_model(X_train, y_train, X_test, y_test, clf)
    filename = f"{name}_content_moderation.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(trained_model, file)
    print(f"Model {name} saved as {filename}\n")

print("Training Done Successfully...")
