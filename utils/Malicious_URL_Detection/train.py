# Importing dependencies
import pickle
import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, auc, roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import math
from urllib.parse import urlparse
from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
# from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import os
import socket
import whois
from datetime import datetime
import time
from bs4 import BeautifulSoup
import urllib
import bs4
import os
import re
from tld import get_tld

df = pd.read_csv('../../data/malicious_phish.csv')

print("Done uploading malicious_phish dataset")


def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        # IPv4 in hexadecimal
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0


df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i))


def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        return 1
    else:
        return 0


df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))

# Feature engineering

df['count.'] = df['url'].apply(lambda i: i.count('.'))

# Count occurrences of 'www'
df['count-www'] = df['url'].apply(lambda i: i.count('www'))

# Count occurrences of '@'
df['count@'] = df['url'].apply(lambda i: i.count('@'))

# Count number of directories in the path component


def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')


df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))


def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')


df['count_embed_domain'] = df['url'].apply(lambda i: no_of_embed(i))

# Check for URL shortening service


def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0


df['short_url'] = df['url'].apply(lambda i: shortening_service(i))

# Count occurrences of 'https'
df['count-https'] = df['url'].apply(lambda i: i.count('https'))

# Count occurrences of 'http'
df['count-http'] = df['url'].apply(lambda i: i.count('http'))

# Count occurrences of '%'
df['count%'] = df['url'].apply(lambda i: i.count('%'))

# Count occurrences of '?'
df['count?'] = df['url'].apply(lambda i: i.count('?'))

# Count occurrences of '-'
df['count-'] = df['url'].apply(lambda i: i.count('-'))

# Count occurrences of '='
df['count='] = df['url'].apply(lambda i: i.count('='))

# Length of URL
df['url_length'] = df['url'].apply(lambda i: len(str(i)))

# Hostname Length
df['hostname_length'] = df['url'].apply(lambda i: len(urlparse(i).netloc))


def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0


df['sus_url'] = df['url'].apply(lambda i: suspicious_words(i))

# First Directory Length


def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0


# Adding 'fd_length' column
df['fd_length'] = df['url'].apply(lambda i: fd_length(i))

# Length of Top Level Domain
df['tld'] = df['url'].apply(lambda i: get_tld(i, fail_silently=True))

# Function to calculate length of TLD


def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1


# Adding 'tld_length' column
df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


# Adding a new column 'count-digits' to the DataFrame
df['count-digits'] = df['url'].apply(lambda i: digit_count(i))


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters


df['count-letters'] = df['url'].apply(lambda i: letter_count(i))

df = df.drop("tld", axis=1)

print("Inforamtion about data after feature engineering:")
print(df.info())


# Model training

lb_make = LabelEncoder()
df["type_code"] = lb_make.fit_transform(df["type"])

# Predictor Variables
X = df[['use_of_ip', 'abnormal_url', 'count.', 'count-www', 'count@',
        'count_dir', 'count_embed_domain', 'short_url', 'count-https',
        'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
        'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
        'count-letters']]

# Target Variable
y = df['type_code']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)


model = xgb.XGBClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

score = metrics.accuracy_score(y_test, y_pred)
print("accuracy of XGBClassifier :   %0.3f" % score)

# Save the model to a file
with open('mul_url_xgb.pkl', 'wb') as f:
    pickle.dump(model, f)
    print("Done saving mul_url_xgb.pkl model")

print("Training Done Successfully ")
