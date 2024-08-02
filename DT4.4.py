# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

# Code to read the dataset and take a quick look
df = pd.read_csv("diabetes.csv")
df.head()

# Investigate the response variable for data imbalance
count0, count1 = df['Outcome'].value_counts()

print(f'The percentage of diabetics in the dataset is only {100*count1/(count0+count1):.2f}%')

# Assign the predictor and response variables
# "Outcome" is the response and all the other columns are the predictors
# Use the values of these features and response
X = df.drop("Outcome",axis=1)
y = df["Outcome"]

# Fix a random_state
random_state = 22

# Split the data into train and validation sets
# Set random state as defined above and use a train size of 0.8
X_train, X_val, y_train, y_val = train_test_split(X,y, train_size=0.8, random_state=random_state)

# Set the max_depth variable to 20 for all trees
max_depth = 20

# Define a Random Forest classifier with randon_state as above
# Set the maximum depth to be max_depth and use 10 estimators
forest = RandomForestClassifier(max_depth=max_depth, n_estimators=10, random_state=random_state)

# Fit the model on the training set
forest.fit(X_train,y_train)

### edTest(test_vanilla) ### 
# Use the trained model to predict on the validation set 
predictions = forest.predict(X_val)

# Compute two metrics that better represent misclassification of minority classes 
# i.e `F1 score` and `AUC`

# Compute the F1-score and assign it to variable score1
f_score = f1_score(y_val, predictions)
score1 = round(f_score, 2)

# Compute the AUC and assign it to variable auc1
auc_score = roc_auc_score(y_val, predictions)
auc1 = round(auc_score, 2)

# Define a Random Forest classifier with randon_state as above
# Set the maximum depth to be max_depth and use 10 estimators
# Use class_weight as balanced_subsample to weigh the class accordingly
random_forest = RandomForestClassifier(max_depth=max_depth, n_estimators=10, random_state=random_state)

# Fit the model on the training set
random_forest.fit(X_train,y_train)

### edTest(test_balanced) ###

# Use the trained model to predict on the validation set 
predictions = random_forest.predict(X_val)

# Compute two metrics that better represent misclassification of minority classes 
# i.e `F1 score` and `AUC`

# Compute the F1-score and assign it to variable score2
f_score = f1_score(y_val, predictions)
score2 = round(f_score, 2)

# Compute the AUC and assign it to variable auc2
auc_score = roc_auc_score(y_val, predictions)
auc2 = round(auc_score, 2)

# Perform upsampling using SMOTE

# Define a SMOTE with random_state=2
sm = SMOTE(random_state=2)

# Use the SMOTE object to upsample the train data
# You may have to use ravel() 
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

# Define a Random Forest classifier with randon_state as above
# Set the maximum depth to be max_depth and use 10 estimators
# Use class_weight as balanced_subsample to weigh the class accordingly
random_forest = RandomForestClassifier(max_depth=max_depth, n_estimators=10, random_state=random_state)

# Fit the Random Forest on upsampled data 
random_forest.fit(X_train_res, y_train_res)

### edTest(test_upsample) ###

# Use the trained model to predict on the validation set 
predictions = random_forest.predict(X_val)

# Compute the F1-score and assign it to variable score3
f_score = f1_score(y_val, predictions)
score3 = round(f_score, 2)

# Compute the AUC and assign it to variable auc3
auc_score = roc_auc_score(y_val, predictions)
auc3 = round(auc_score, 2)

# Define an RandomUnderSampler instance with random state as 2
rs = RandomUnderSampler(random_state=2)

# Downsample the train data
# You may have to use ravel()
X_train_res, y_train_res = rs.fit_resample(X_train_res, y_train_res.ravel())

# Define a Random Forest classifier with randon_state as above
# Set the maximum depth to be max_depth and use 10 estimators
# Use class_weight as balanced_subsample to weigh the class accordingly
random_forest = RandomForestClassifier(max_depth=max_depth, n_estimators=10, random_state=random_state)

# Fit the Random Forest on downsampled data 
random_forest.fit(X_train_res, y_train_res)

### edTest(test_downsample) ###

# Use the trained model to predict on the validation set 
predictions = random_forest.predict(X_val)

# Compute two metrics that better represent misclassification of minority classes 
# i.e `F1 score` and `AUC`

# Compute the F1-score and assign it to variable score4
f_score = f1_score(y_val, predictions)
score4 = round(f_score, 2)

# Compute the AUC and assign it to variable auc4
auc_score = roc_auc_score(y_val, predictions)
auc4 = round(auc_score, 2)

# Compile the results from the implementations above

pt = PrettyTable()
pt.field_names = ["Strategy","F1 Score","AUC score"]
pt.add_row(["Random Forest - No imbalance correction",score1,auc1])
pt.add_row(["Random Forest - balanced_subsamples",score2,auc2])
pt.add_row(["Random Forest - Upsampling",score3,auc3])
pt.add_row(["Random Forest - Downsampling",score4,auc4])
print(pt)

