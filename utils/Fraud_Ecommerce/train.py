import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import calendar
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve, confusion_matrix, classification_report

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

# Predict probabilities for class 1 (Fraud)
y_prob = logistic_regression.predict_proba(X_test)[:, 1]

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Fraud_Ecommerce (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()

# Save ROC curve plot
plt.savefig('results/roc_curve.png')
plt.close()

# Print AUC score
print("AUC:", roc_auc)

# Confusion matrix
y_pred = logistic_regression.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df.to_csv('results/classification_report.csv')

# Save confusion matrix plot
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])
plt.yticks([0, 1], ['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png')
plt.close()

# Save classification report to a text file
with open('results/classification_report.txt', 'w') as f:
    f.write(classification_report(y_test, y_pred))
