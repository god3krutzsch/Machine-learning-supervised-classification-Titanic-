# try to build a learning model that predicts the type of people that survived the titanic
# import the training data first

import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load training and test data

file_path = "/Users/godfreykrutzsch/Desktop/ML-Challenge/train.csv"
training_data = pd.read_csv(file_path)
file_path_test = "/Users/godfreykrutzsch/Desktop/ML-Challenge/test.csv"
test_data = pd.read_csv(file_path_test)

# exploratory data analysis first check for missing values across the entire training data.
# some inconsistencies with fare and embarked for training and test data.

print("training data head")
print(training_data.head())
print("training data info")
print(training_data.info())

print("look at missing values training data")
print(training_data.isnull().sum())
print("look at missing values test data")
print(test_data.isnull().sum())
print(training_data.describe())

# preprocessing extract the target variable this is a value 1 survived 0 did not survive
y = training_data['Survived']

# we should remove cabin and ticket as will not affect the prediction
training_data.drop('Cabin', axis=1, inplace=True)
training_data.drop('Ticket', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Ticket', axis=1, inplace=True)

# remember to drop survived from training or it will be treated as a feature and expected in the test data
training_data.drop('Survived', axis=1, inplace=True)

# fill missing values
training_data.fillna({'Age': training_data['Age'].mean()}, inplace=True)
training_data.fillna({'Fare': training_data['Fare'].mean()}, inplace=True)
test_data.fillna({'Age': test_data['Age'].mean()}, inplace=True)
test_data.fillna({'Fare': test_data['Fare'].mean()}, inplace=True)

# check fields to ensure no blanks or NAN
print(training_data.isnull().sum())

# one hot encoding
training_data = pd.get_dummies(training_data, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']) * 1
test_data = pd.get_dummies(test_data, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']) * 1

title_integers = []
titles = []

# we must extricate the name from the freetext within the training dataframe the title e.g. miss, master, rev, mr
# and assign for each a unique integer so we can train the model with this new feature.
# look out for a better method. Capt and Don are not in test data so do not train the AI on these labels

for name in training_data['Name']:
    # Split the name into words
    words = name.split()
    # print(words)
    # Check if 'Mrs' is in the words
    if 'Mrs.' in words:
        titles.append("Mrs.")
        title_integers.append(1)
    #  Check if 'Miss' is in the words
    elif 'Miss.' in words:
        titles.append('Miss.')
        title_integers.append(2)
    elif 'Mr.' in words:
        titles.append('Mr')
        title_integers.append(3)
    elif 'Rev.' in words:
        titles.append('Rev.')
        title_integers.append(4)
    else:
        # to avoid a dimension problem we catch unbekannt or the number of titles will not add up to the
        # number of rows in the training data, therefore we can not add the new columns title or title_integers
        # to the dataframe.

        titles.append('Unbekannt')
        title_integers.append(0)

training_data['Title'] = titles
training_data['Title_integers'] = title_integers

# Hot encoding for title
training_data = pd.get_dummies(training_data, columns=['Title', 'Title_integers']) * 1

# we cannot use this field for machine learning
training_data.drop('Name', axis=1, inplace=True)

# iteration warning
logr = linear_model.LogisticRegression(max_iter=2000)

x = training_data

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.5,random_state=1)


print("start training AI")
print(y_train)
logr.fit(X_train, y_train)
training_score = logr.predict(X_test)
print("Training score")
print(training_score)
# test data

titles_test = []
title_integers_test = []

# Capt. and Don are not in the test data so do not train the AI to learn these labels lucky i went through both

for name in test_data['Name']:
    # Split the name into words
    words = name.split()
    if 'Mrs.' in words:
        titles_test.append("Mrs.")
        title_integers_test.append(1)
    # Check if 'Miss' is in the words
    elif 'Miss.' in words:
        titles_test.append('Miss.')
        title_integers_test.append(2)
    elif 'Mr.' in words:
        titles_test.append('Mr')
        title_integers_test.append(3)
    elif 'Rev.' in words:
        titles_test.append('Rev.')
        title_integers_test.append(4)
    else:
        titles_test.append('Unbekannt')
        title_integers_test.append(0)

test_data['Title'] = titles_test
test_data['Title_integers'] = title_integers_test

# hot encoding
test_data = pd.get_dummies(test_data, columns=['Title', 'Title_integers']) * 1

# drop name as cannot be used for ML and dropped in training
test_data.drop('Name', axis=1, inplace=True)

# drop as this category does not exist in the limited test data but does in training date..
test_data.drop('Parch_9', axis=1, inplace=True)

# make prediction

test_feature = test_data
predicted = logr.predict(test_feature)

print("Prediction")
print(predicted)

count_of_ones = np.count_nonzero(predicted)

print("The number of survivors")
print(count_of_ones)
print("The number of dead")
count_of_zeros = np.count_nonzero(predicted == 0)
print(count_of_zeros)


# evaluate the score

print(logr.score(X_train, y_train))



# Create the submission file

# Create a DataFrame with 'PassengerId' as index and 'Survived' as a column tried fitting hyperparameters without any impact
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predicted  # Replace None with the data you want to use for the 'Survived' column
})

# Save the DataFrame to a CSV file
submission.to_csv('/Users/godfreykrutzsch/Desktop/ML-Challenge/titanic-no-drop-1.csv',
                  index=False)  # Set index=False if you don't want to include the DataFrame index
