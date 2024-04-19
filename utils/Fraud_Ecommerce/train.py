import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import calendar

# Read the datasets
user_info = pd.read_csv("../../data/new_fraud.csv")
print(user_info.info())

print("Done uploading dataset ..")

user_info["purchase_time"] = pd.to_datetime(user_info["purchase_time"])
user_info["signup_time"] = pd.to_datetime(user_info["signup_time"])

user_info["month_purchase"] = user_info.purchase_time.apply(
    lambda x: calendar.month_name[x.month])

# Column week
user_info["weekday_purchase"] = user_info.purchase_time.apply(
    lambda x: calendar.day_name[x.weekday()])


# Column hour_of_the_day
user_info["hour_of_the_day"] = user_info.purchase_time.apply(lambda x: x.hour)

# Hour of the day categorisation
user_info["period_of_the_day"] = user_info.hour_of_the_day.apply(lambda x:
                                                                 "late night" if x < 4 else
                                                                 "early morning" if x < 8 else
                                                                 "morning" if x < 12 else
                                                                 "early arvo" if x < 16 else
                                                                 "arvo" if x < 20 else
                                                                 "evening"
                                                                 )
# age categorisation
user_info["age_category"] = user_info.age.apply(lambda x:
                                                "< 40" if x < 40 else
                                                "40 - 49" if x < 50 else
                                                "50 -59" if x < 60 else
                                                "60 - 69" if x < 70 else
                                                " > 70")

user_info["seconds_since_signup"] = (
    user_info.purchase_time - user_info.signup_time).apply(lambda x: x.total_seconds())


print(user_info.info())
pass


# Column "quick_purchase" : categorise time between sign_up and purchase
user_info["quick_purchase"] = user_info.seconds_since_signup.apply(
    lambda x: 1 if x < 30 else 0)


# Define the list of columns to remove
columns_to_remove = ["user_id", "signup_time", "purchase_time", "device_id",
                     "ip_address", "hour_of_the_day", "seconds_since_signup", "age"]

# Drop the specified columns from the DataFrame
user_info.drop(columns_to_remove, axis=1, inplace=True)

# Drop rows with missing values from the user_info dataset
user_info.dropna(inplace=True)


# Data preprocessing
features = pd.get_dummies(user_info.drop("class", axis=1), drop_first=True)
target = user_info["class"]

# Save column names for future use
train_columns = features.columns.tolist()

# Save train_columns
with open("train_columns.pkl", "wb") as f:
    pickle.dump(train_columns, f)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    random_state=42,
                                                    stratify=target,
                                                    test_size=0.25)

# Fitting a logistic regression model
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train, y_train)

# Printing scores
train_score = logistic_regression.score(X_train, y_train)
test_score = logistic_regression.score(X_test, y_test)
print("Train Score:", round(train_score * 100, 2), "%")
print("Test Score:", round(test_score * 100, 2), "%")

# Saving the trained model using pickle
with open("fraudE_log_reg.pkl", "wb") as f:
    pickle.dump(logistic_regression, f)
