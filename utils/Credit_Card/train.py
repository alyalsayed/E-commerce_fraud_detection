#!/usr/bin/env python
# coding: utf-8

# # Fraud detection

# ## Fraud_Data DataFrame:
# * ## user_id: Unique user identifier.
# * ## signup_time: Timestamp of user sign-up.
# * ## purchase_time: Timestamp of purchase.
# * ## purchase_value: Value of the purchase.
# * ## device_id: Unique device identifier.
# * ## source: Source of user arrival.
# * ## browser: Web browser used by the user.
# * ## sex: Gender of the user.
# * ## age: Age of the user.
# * ## ip_address: User's IP address.
# * ## class: Transaction class (0 for non-fraudulent, 1 for fraudulent).
#  
# ## ip_country_mapping DataFrame:
# * ## lower_bound_ip_address: Lower bound of IP address range.
# * ## upper_bound_ip_address: Upper bound of IP address range.
# * ## country: Country associated with the IP address range.

# In[1]:


import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
import datetime
import calendar
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Read the datasets
user_info = pd.read_csv("../input/fraud-ecommerce/Fraud_Data.csv")         # Users information
ip_country_mapping = pd.read_csv("../input/fraud-ecommerce/IpAddress_to_Country.csv")  # Country from IP information


# In[2]:


ip_country_mapping.head()


# In[3]:


ip_country_mapping.info()


# In[4]:


user_info.head()


# In[5]:


user_info.info()


# In[6]:


ip_country_mapping.upper_bound_ip_address.astype("float")
ip_country_mapping.lower_bound_ip_address.astype("float")
user_info.ip_address.astype("float")


# In[7]:


def IP_to_country(ip) :
    try :
        return ip_country_mapping.country[(ip_country_mapping.lower_bound_ip_address < ip)                            
                                & 
                                (ip_country_mapping.upper_bound_ip_address > ip)].iloc[0]
    except IndexError :
        return "Unknown"   


# In[8]:


import os

# Define the directory path
directory = "/kaggle/working/datasets_fraud"

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)
    
# country to each IP
user_info["IP_country"] = user_info.ip_address.apply(IP_to_country)

# saving
user_info.to_csv("/kaggle/working/datasets_fraud/Fraud_data_with_country.csv",index=False)


# In[9]:


# loading
user_info= pd.read_csv("/kaggle/working/datasets_fraud/Fraud_data_with_country.csv")

user_info.head()


# In[10]:


def IP_to_country2(ip) :
    try :
        return ip_country_mapping.country[(ip_country_mapping.lower_bound_ip_address < ip)                            
                                & 
                                (ip_country_mapping.upper_bound_ip_address > ip)]
    except IndexError :
        return "Unknown"     
    
print(IP_to_country2(user_info.iloc[0]['ip_address']))
print(IP_to_country2(user_info.iloc[1]['ip_address']))


# In[11]:


# Print summary statistics 
print(user_info[["purchase_value", "age"]].describe())
print('*'*50)
# Print unique values and their frequencies 
for column in ["source", "browser", "sex"]:
    print(user_info[column].value_counts())
    print('*'*50)

# Check for duplicates in the "user_id" column in user_info DataFrame
print("The user_id column includes {} duplicates".format(user_info.duplicated(subset="user_id", keep=False).sum()))


# In[12]:


# Calculate duplicate rate based on unique device_id
dup_table = pd.DataFrame(user_info.duplicated(subset="device_id"))
dup_rate = dup_table.mean()
print("{}% of the dataset is comprised of transactions from a device_id that had been previously used".format(int(dup_rate * 1000) / 10))

# Calculate duplicate rate based on device_id with keep=False
dup_table2 = pd.DataFrame(user_info.duplicated(subset="device_id", keep=False))
dup_rate2 = dup_table2.mean()
print("{}% of the dataset is comprised of transactions from a device_id that had been previously used".format(int(dup_rate2 * 1000) / 10))


# In[13]:


device_duplicates = pd.DataFrame(user_info.groupby(by="device_id").device_id.count())
device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)
device_duplicates.reset_index(level=0, inplace=True)
dupli = device_duplicates[device_duplicates.freq_device >1]
dupli


# In[14]:


# Reading the Dataset
user_info = pd.read_csv("/kaggle/working/datasets_fraud/Fraud_data_with_country.csv")

device_duplicates = pd.DataFrame(user_info.groupby(by = "device_id").device_id.count())  
device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)           
device_duplicates.reset_index(level=0, inplace= True)                                 

dupli = device_duplicates[device_duplicates.freq_device >1]
print("On average, when a device is used more than once it is used {mean} times, and the most used machine was used {maxi} times"
      .format(mean = int(dupli.freq_device.mean()*10)/10, maxi = int(dupli.freq_device.max()*10)/10))

dupli = device_duplicates[device_duplicates.freq_device >2]
print("On average, when a device is used more than twice it is used {mean} times"
      .format(mean = int(dupli.freq_device.mean()*10)/10, maxi = int(dupli.freq_device.max()*10)/10))


# In[15]:


# Merge the device_duplicates with user_info
user_info = user_info.merge(device_duplicates, on="device_id")


# In[16]:


# Calculate the proportion of fraud in the dataset
fraud_proportion = user_info["class"].mean() * 100
print("Proportion of fraud in the dataset: {:.1f}%".format(fraud_proportion))


# In[17]:


user_info.describe()


# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots
f, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot device frequency distribution for values less than 4
g1 = sns.distplot(user_info.freq_device[user_info.freq_device < 4], ax=ax[0])
g1.set(xticks=[1, 2, 3])

# Plot device frequency distribution for values greater than 2
g2 = sns.distplot(user_info.freq_device[user_info.freq_device > 2], ax=ax[1])
g2.set(xticks=range(0, 21, 2))

# Display the plots
plt.show()


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(18, 6))

# Create subplots for bar plots
plt.subplot(1, 3, 1)
sns.barplot(x='source', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Source')

plt.subplot(1, 3, 2)
sns.barplot(x='browser', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Browser')

plt.subplot(1, 3, 3)
sns.barplot(x='sex', y='class', data=user_info, ci=None)
plt.title('Fraud Proportion by Sex')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplots
f2, ax2 = plt.subplots(3, 1, figsize=(24, 18))

# Plot purchase_value vs. class
sns.pointplot(x="purchase_value", y="class", data=user_info, ci=None, ax=ax2[0])
ax2[0].set_title("Purchase Value vs. Fraud Probability")

# Plot age vs. class
sns.pointplot(x="age", y="class", data=user_info, ci=None, ax=ax2[1])
ax2[1].set_title("Age vs. Fraud Probability")

# Plot freq_device vs. class
sns.pointplot(x="freq_device", y="class", data=user_info, ci=None, ax=ax2[2])
ax2[2].set_title("Frequency of Device Usage vs. Fraud Probability")

# Show the plots
plt.tight_layout()
plt.show()


# In[21]:


user_info.head()


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a figure and axis
f3, ax3 = plt.subplots(1, 1, figsize=(24, 18))

# Plot a stacked bar plot for IP_country vs. class
sns.barplot(x="IP_country", y="class", data=user_info[:10], estimator=sum, ci=None, ax=ax3)

# Show the plot
plt.show()


# In[23]:


# Filter IP_country value counts where count is greater than 1000
filtered_counts = user_info.IP_country.value_counts()[user_info.IP_country.value_counts() > 1000]

# Plot the filtered counts as a bar plot
filtered_counts.plot(kind="bar")
plt.xlabel("IP Country")
plt.ylabel("Frequency")
plt.title("IP Country Frequency (Counts > 1000)")
plt.show()


# In[24]:


user_info.signup_time


# ## Feature engineering

# In[25]:


# --- 1 ---
# Categorisation column freq_device
# We see a clear correlation between freq_device and fraudulent activities. We are going to split freq_device into 7 categories
user_info.freq_device = user_info.freq_device.apply(lambda x:
                                                    str(x) if x < 5 else
                                                    "5-10" if x >= 5 and x <= 10 else
                                                    "11-15" if x > 10 and x <= 15 else
                                                    "> 15")


# In[26]:


# Convert signup_time and purchase_time to datetime
user_info.signup_time = pd.to_datetime(user_info.signup_time, format='%Y-%m-%d %H:%M:%S')
user_info.purchase_time = pd.to_datetime(user_info.purchase_time, format='%Y-%m-%d %H:%M:%S')


# In[27]:


# --- 2 ---
# Column month
user_info["month_purchase"] = user_info.purchase_time.apply(lambda x: calendar.month_name[x.month])

# --- 3 ---
# Column week
user_info["weekday_purchase"] = user_info.purchase_time.apply(lambda x: calendar.day_name[x.weekday()])

# --- 4 ---
# Column hour_of_the_day
user_info["hour_of_the_day"] = user_info.purchase_time.apply(lambda x: x.hour)

# --- 5 ---
# Column seconds_since_signup
user_info["seconds_since_signup"] = (user_info.purchase_time - user_info.signup_time).apply(lambda x: x.total_seconds())


# In[28]:


# --- 6 ---
# Column countries_from_device (ie. number of different countries per device_id)
# We flag devices that committed purchases from different countries
country_count = user_info.groupby(by=["device_id", "IP_country"]).count().reset_index()
country_count = pd.DataFrame(country_count.groupby(by="device_id").count().IP_country)
user_info = user_info.merge(country_count, left_on="device_id", right_index=True)
user_info.rename(columns={"IP_country_x": "IP_country", "IP_country_y": "countries_from_device"}, inplace=True)


# In[29]:


user_info.head()


# In[30]:


# Step 1: Calculate the proportion of fraudulent transactions for each country
fraud_rate_by_country = user_info.groupby('IP_country')['class'].mean().sort_values(ascending=False)

# Step 2: Categorize countries based on their fraud rates into risk levels
risk_levels = pd.cut(fraud_rate_by_country, bins=[-np.inf, 0.01, 0.05, 0.25, np.inf], labels=['Low risk', 'Medium risk', 'High risk', 'Very high risk'])

# Combine the results into a DataFrame
risk_country = pd.DataFrame({'fraud_rate': fraud_rate_by_country, 'risk_level': risk_levels})
risk_country


# In[31]:


top_10_countries = risk_country.head(10)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_10_countries.index, y='fraud_rate', data=top_10_countries, hue='risk_level', dodge=False)
plt.xticks(rotation=45)
plt.xlabel('Country')
plt.ylabel('Fraud Rate')
plt.title('Top 10 Countries with Highest Fraud Rates and Their Risk Levels')
plt.legend(title='Risk Level')
plt.show()


# In[32]:


user_info = user_info.merge(risk_country, left_on="IP_country", right_index=True)


# In[33]:


# --- 8 ---
# Column "quick_purchase" : categorise time between sign_up and purchase
user_info["quick_purchase"] = user_info.seconds_since_signup.apply(lambda x: 1 if x < 30 else 0)

# --- 9 ---
# Column freq_same_purchase : indicates how many times a given device_id purchased an item of the same value
duplicate = user_info.duplicated(subset=["purchase_value", "device_id"], keep=False)
duplicate = pd.concat([user_info.loc[:, ["purchase_value", "device_id"]], duplicate], axis=1)
duplicate = duplicate.groupby(by=["device_id", "purchase_value"]).sum()
duplicate["freq_same_purchase"] = duplicate[0].apply(lambda x:
                                                      x if x < 5 else
                                                      "5-10" if x <= 10 else
                                                      "11-15" if x <= 15 else
                                                      ">15"
                                                      )
user_info = user_info.merge(duplicate.drop(0, axis=1), left_on=["device_id", "purchase_value"], right_index=True)


# In[34]:


# --- 10 ----
# age categorisation
user_info["age_category"] = user_info.age.apply(lambda x:
                                                "< 40" if x < 40 else
                                                "40 - 49" if x < 50 else
                                                "50 -59" if x < 60 else
                                                "60 - 69" if x < 70 else
                                                " > 70")



# In[35]:


# ---- 11 ----
# Hour of the day categorisation
user_info["period_of_the_day"] = user_info.hour_of_the_day.apply(lambda x:
                                                                 "late night" if x < 4 else
                                                                 "early morning" if x < 8 else
                                                                 "morning" if x < 12 else
                                                                 "early arvo" if x < 16 else
                                                                 "arvo" if x < 20 else
                                                                 "evening"
                                                                 )


# In[36]:


# ---- 12 -----
# First_purchase 
#this cell took long time 
user_info["first_purchase"] = user_info.apply(lambda x : 
                                         1 if x.purchase_time == user_info.purchase_time[user_info.device_id == x.device_id].min() else 0,
                                         axis =1)

user_info.first_purchase.to_csv("/kaggle/working/datasets_fraud/Series_first_purchase.csv",index =False)



# In[37]:


user_info["first_purchase"] = pd.read_csv("/kaggle/working/datasets_fraud/Series_first_purchase.csv" )

user_info.to_csv("/kaggle/working/datasets_fraud/data_with_first_feature_eng.csv")


# In[38]:


import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn
import datetime
import calendar
import warnings


# In[39]:


user_info=pd.read_csv("/kaggle/input/fraud-cleaned/datasets_fraud/data_with_first_feature_eng.csv")


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns

liste_col = ["freq_device", "month_purchase", "weekday_purchase",
             "countries_from_device", "risk_level", "quick_purchase",
             "age_category", "period_of_the_day"]

# Define the order for categorical variables
param_order = {
    "freq_device": ["1", "2", "3", "4", "5-10", "11-15", "> 15"],
    "month_purchase": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "December"],
    "weekday_purchase": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "risk_level": ["Low risk", "Medium risk", "High risk", "Very High risk"],
    "period_of_the_day": ["morning", "early arvo", "arvo", "evening", "late night", "early morning"]
}

# Create subplots
fig, axes = plt.subplots(len(liste_col), 1, figsize=(20, 30))

# Iterate through each column and create a pointplot
for i, col in enumerate(liste_col):
    sns.pointplot(x=col, y="class", data=user_info, order=param_order.get(col), ax=axes[i])
    axes[i].set_xlabel(col)

plt.tight_layout()
plt.show()


# In[41]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define the columns and their order
liste_col = ["freq_device", "month_purchase", "weekday_purchase",
             "countries_from_device", "risk_level", "quick_purchase",
             "age_category", "period_of_the_day"]

# Define the order for categorical variables
param_order = {
    "freq_device": ["1", "2", "3", "4", "5-10", "11-15", "> 15"],
    "month_purchase": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "December"],
    "weekday_purchase": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "risk_level": ["Low risk", "Medium risk", "High risk", "Very High risk"],
    "period_of_the_day": ["morning", "early arvo", "arvo", "evening", "late night", "early morning"]
}

# Create subplots
fig, axes = plt.subplots(len(liste_col), 1, figsize=(20, 30))

# Iterate through each column and create a stacked bar plot
for i, col in enumerate(liste_col):
    sns.countplot(x=col, hue="class", data=user_info, order=param_order.get(col), ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].legend(title='class')

plt.tight_layout()
plt.show()


# In[42]:


import pandas as pd

# Function for moving a column to the recycling dataset
def to_recycle_bin(column):
    recycle_ds[column] = user_info[column]
    user_info.drop(column, axis=1, inplace=True)

# Set the maximum number of columns to display
pd.set_option('display.max_columns', 500)

# Define the list of columns to remove
columns_to_remove = ["user_id", "signup_time", "purchase_time", "device_id", "ip_address", "IP_country", "hour_of_the_day", "seconds_since_signup", "age"]

# Initialize the recycling dataset
recycle_ds = pd.DataFrame()

# Loop through the columns and move them to the recycling dataset
for column in columns_to_remove:
    to_recycle_bin(column)

# Drop rows with missing values from the user_info dataset
user_info.dropna(inplace=True)


# In[43]:


user_info["class"].value_counts()


# ## Logistic regression

# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Data preprocessing
features = pd.get_dummies(user_info.drop("class", axis=1), drop_first=True)
target = user_info["class"]

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    random_state=42,
                                                    stratify=target,
                                                    test_size=0.25)

# Normalizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fitting a logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Printing scores
train_score = logistic_regression.score(X_train, y_train)
test_score = logistic_regression.score(X_test, y_test)
print("Train Score:", round(train_score * 100, 2), "%")
print("Test Score:", round(test_score * 100, 2), "%")


# In[45]:


# Setting up probability threshold
prob_threshold = 0.22

# Predicting probabilities and applying threshold
probabilities = pd.DataFrame(logistic_regression.predict_proba(X_test), columns=["prob_no_fraud", "prob_fraud"]).drop("prob_no_fraud", axis=1)
predictions = probabilities.prob_fraud.apply(lambda x: 0 if x < prob_threshold else 1)

# Calculating score with threshold
threshold_score = np.mean(predictions.reset_index(drop=True) == y_test.reset_index(drop=True))

# Printing results with threshold
print("Predicted fraud count:", sum(predictions),
      "with a threshold of", prob_threshold * 100, "%",
      "Test Score with threshold:", round(threshold_score * 100, 2), "%")

# Confusion matrix
f, ax = plt.subplots(2, 1, figsize=(8, 8))

# Confusion matrix with default threshold (50%)
cm_default = confusion_matrix(y_test, logistic_regression.predict(X_test))
sns.heatmap(cm_default, annot=True, fmt="d", ax=ax[0])
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('True')

# Confusion matrix with personalized threshold
cm_threshold = confusion_matrix(y_test, predictions)
sns.heatmap(cm_threshold, annot=True, fmt="d", ax=ax[1])
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('True')

plt.show()


# In[46]:


from sklearn.metrics import roc_auc_score, roc_curve

# Calculate ROC AUC score
logit_roc_auc = roc_auc_score(y_test, logistic_regression.predict(X_test))

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, logistic_regression.predict_proba(X_test)[:,1])

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[47]:


import joblib
from sklearn.metrics import accuracy_score
# Save the model to a file
joblib.dump(logistic_regression, 'ecommerce_log_reg.pkl')


# In[48]:


#  load the model from the file
loaded_model = joblib.load('ecommerce_log_reg.pkl')

# use the loaded model to make predictions
predicted_classes = loaded_model.predict(X_test)


# In[49]:


# Calculate accuracy score
accuracy = accuracy_score(y_test, predicted_classes)
print("Accuracy:", accuracy) 


# ## Logistic regression with SMOTE

# In[50]:


from imblearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Define the resampling method and the ML model
resampling = BorderlineSMOTE()
model = LogisticRegression(solver='liblinear')

# Define the pipeline with resampling and the model
pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

# Fit the pipeline onto the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
predicted = pipeline.predict(X_test)

# Print the classification report and confusion matrix
print('Classification report:\n', classification_report(y_test, predicted))
print('Confusion matrix:\n', confusion_matrix(y_test, predicted))

# Calculate and print accuracy scores
train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[51]:


# Save the resampling method and the model separately
joblib.dump(resampling, 'resampling_model.pkl')
joblib.dump(model, 'logistic_regression_model.pkl')


# In[52]:


# Load the resampling method and the model
resampling = joblib.load('resampling_model.pkl')
model = joblib.load('logistic_regression_model.pkl')

# Recreate the pipeline
pipeline = Pipeline([('SMOTE', resampling), ('Logistic Regression', model)])

train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[53]:


from sklearn.ensemble import RandomForestClassifier
Random_forest_model = RandomForestClassifier(random_state=5, n_estimators=20)
Random_forest_model.fit(X_train, y_train)
train_accuracy = Random_forest_model.score(X_train, y_train)
test_accuracy = Random_forest_model.score(X_test, y_test)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[54]:


joblib.dump(Random_forest_model, 'ecommerce_Random_forest_model.pkl')


# # fraud in Credit Card

# ## DataFrame Description:
# * ## Time: Timestamp of the transaction.
# * ## V1-V28: Features generated by PCA transformation to protect user identities and sensitive information.
# * ## Amount: Transaction amount.
# * ## Class: Class label indicating whether the transaction is fraudulent (1) or not (0).

# In[55]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[56]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')


# In[57]:


# first 5 rows of the dataset
credit_card_data.head()


# In[58]:


credit_card_data.info()


# In[59]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[60]:


credit_card_data['Class'].value_counts()


# This Dataset is highly unblanced
# 
# 0 --> Normal Transaction
# 
# 1 --> fraudulent transaction

# In[61]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[62]:


print(legit.shape)
print(fraud.shape)


# In[63]:


# statistical measures of the data
legit.Amount.describe()


# In[64]:


fraud.Amount.describe()


# In[65]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# Under-Sampling
# 
# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
# 
# Number of Fraudulent Transactions --> 492

# In[66]:


legit_sample = legit.sample(n=492)


# In[67]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[68]:


new_dataset.head()


# In[69]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[70]:


print(X)


# In[71]:


print(Y)


# In[72]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)


# In[73]:


print(X.shape, X_train.shape, X_test.shape)


# In[74]:


print(Y.shape, Y_train.shape, Y_test.shape)


# ## Model Training

# In[75]:


model = LogisticRegression()


# In[76]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# In[77]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[78]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[79]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[80]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# In[81]:


import pickle

# Save the model to a file
with open('credit_card_log_reg.pkl', 'wb') as file:
    pickle.dump(model, file)


# # Malicious URL Detection

# ## DataFrame Description:
# 
# *This DataFrame contains website URLs along with their corresponding types.*
# 
# *  url: The URL of the website.
# *  type: The type of website, categorized into various types such as phishing, benign, defacement, etc.

# In[82]:


get_ipython().system('pip install python-whois')


# In[83]:


import pandas as pd
import itertools
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import math
from collections import Counter
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import xgboost as xgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import os
import socket
import whois
from datetime import datetime
import time
from bs4 import BeautifulSoup
import urllib
import bs4
import os


# In[84]:


df=pd.read_csv('/kaggle/input/malicious-urls-dataset/malicious_phish.csv')

print(df.shape)
df.head()


# In[85]:


df["type"].value_counts()


# ## Feature Engineering
# 

# In[86]:


import re

def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0

df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i))
print(df.head())


# In[87]:


df.use_of_ip.value_counts()


# In[88]:


from urllib.parse import urlparse
import re

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        return 1
    else:
        return 0

df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))
df.head()


# In[89]:


print(df.url.iloc[3])
abnormal_url(df["url"][3])


# In[90]:


df.abnormal_url.value_counts()


# In[91]:


get_ipython().system('pip install googlesearch-python')


# ## Feature engineering

# In[92]:


df['count.'] = df['url'].apply(lambda i: i.count('.'))
df.head()


# In[93]:


df['count.'].value_counts()


# In[94]:


import re
from urllib.parse import urlparse

# Count occurrences of 'www'
df['count-www'] = df['url'].apply(lambda i: i.count('www'))
print(df['count-www'].value_counts())
df.head()


# In[95]:


# Count occurrences of '@'
df['count@'] = df['url'].apply(lambda i: i.count('@'))
print(df['count@'].value_counts())
df.tail()


# In[96]:


# Count number of directories in the path component
def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')
print(df["url"][3])
print(no_of_dir(df["url"][3]))


# In[97]:


df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))
print(df['count_dir'].value_counts())
df.head()


# In[98]:


# Count occurrences of '//'
def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')
print(df["url"][3])
print(no_of_embed(df["url"][3]))


# In[99]:


df['count_embed_domain'] = df['url'].apply(lambda i: no_of_embed(i))
print(df['count_embed_domain'].value_counts())
df.head()


# In[100]:


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

print(df['short_url'].value_counts())
df.head()


# In[101]:


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


# In[102]:


df.head()


# In[103]:


def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0

df['sus_url'] = df['url'].apply(lambda i: suspicious_words(i))

df.tail()


# In[104]:


get_ipython().system('pip install tld')


# In[105]:


# Importing dependencies
from urllib.parse import urlparse
from tld import get_tld

# First Directory Length
def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

print(df.url[8])
print(fd_length(df.url[8]))
print(get_tld(df.url[8], fail_silently=True))


# Explanation:
# 
# The URL **http://www.pashminaonline.com/pure-pashminas** has the following structure: protocol://domain/path. The path component of the URL is /pure-pashminas.
# After splitting the path by /, the first part is pure-pashminas. The length of the first directory is 14 characters (pure-pashminas).
# The top-level domain (TLD) of the URL is com.

# In[106]:


# Adding 'fd_length' column
df['fd_length'] = df['url'].apply(lambda i: fd_length(i))

# Length of Top Level Domain
df['tld'] = df['url'].apply(lambda i: get_tld(i, fail_silently=True))


# In[107]:


# Function to calculate length of TLD
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

# Adding 'tld_length' column
df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))

df.head()


# In[108]:


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

# Adding a new column 'count-digits' to the DataFrame
df['count-digits'] = df['url'].apply(lambda i: digit_count(i))


# In[109]:


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters
df['count-letters']= df['url'].apply(lambda i: letter_count(i))


# In[110]:


df.head()


# In[111]:


df = df.drop("tld", axis=1)


# In[112]:


df.info()


# ## Model training

# In[113]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
df["type_code"] = lb_make.fit_transform(df["type"])
df["type_code"].value_counts()


# In[114]:


# Predictor Variables
X = df[['use_of_ip','abnormal_url', 'count.', 'count-www', 'count@',
        'count_dir', 'count_embed_domain', 'short_url', 'count-https',
        'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
        'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
        'count-letters']]

# Target Variable
y = df['type_code']


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)


# In[116]:


lgb = LGBMClassifier(objective='multiclass',boosting_type= 'gbdt',n_jobs = 5, 
          silent = True, random_state=5)
LGB_C = lgb.fit(X_train, y_train)


y_pred = LGB_C.predict(X_test)
print(classification_report(y_test,y_pred))

score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)


# In[117]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[118]:


cm = metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2,3])
plot_confusion_matrix(cm,classes=['benign', 'defacement','phishing','malware'])


# In[119]:


lgb_feature = lgb.feature_importances_
lgb_feature


# In[120]:


lgb_features = lgb_feature.tolist()
lgb_features


# In[121]:


import pickle

# Save the model to a file
with open('mul_urls_lgb_model.pkl', 'wb') as f:
    pickle.dump(LGB_C, f)


# In[122]:


from sklearn.metrics import accuracy_score
with open('mul_urls_lgb_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)


# In[123]:


model = xgb.XGBClassifier(n_estimators= 100)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))


score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)


# In[124]:


CM=confusion_matrix(y_test,y_pred,labels=[0,1,2,3])

plot_confusion_matrix(cm,classes=['benign', 'defacement','phishing','malware'])


# In[125]:


import pickle

# Save the model to a file
with open('mul_url_xgb.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[126]:


xgb_feature = model.feature_importances_
xgb_features = xgb_feature.tolist()


# In[127]:


from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(n_estimators=100,max_features='sqrt')
gbdt.fit(X_train,y_train)
y_pred = gbdt.predict(X_test)
print(classification_report(y_test,y_pred))

score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)


# In[128]:


CM=confusion_matrix(y_test,y_pred,labels=[0,1,2,3])

plot_confusion_matrix(cm,classes=['benign', 'defacement','phishing','malware'])


# In[129]:


import pickle

# Save the model to a file
with open('mul_url_gbdt.pkl', 'wb') as f:
    pickle.dump(gbdt, f)


# In[130]:


gbdt_feature = gbdt.feature_importances_
gbdt_features = gbdt_feature.tolist()


# In[131]:


print(gbdt_features)
print(xgb_features)
print(lgb_features)


# In[132]:


cols = X_train.columns
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
    
    'Gradient Boost feature importances': gbdt_features,
    'XG Boost feature importances': xgb_features,
    'LGBM feature importances': lgb_features
                                   
    })
feature_dataframe


# In[133]:


# Select only the numeric columns for calculating the mean
numeric_cols = feature_dataframe.columns[1:]  # Exclude the 'features' column
feature_dataframe['mean'] = feature_dataframe[numeric_cols].mean(axis=1)


# In[134]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
def plot_feature_importance():
    tmp = pd.DataFrame({'Feature': X_test.columns, 'Feature importance': feature_dataframe['mean'].values})
    tmp = tmp.sort_values(by='Feature importance',ascending=False).head(20)
    plt.figure(figsize = (10,12))
    plt.title('Average Feature Importance Top 20 Features',fontsize=14)
    s = sns.barplot(y='Feature',x='Feature importance',data=tmp, orient='h')
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()
plot_feature_importance()


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

