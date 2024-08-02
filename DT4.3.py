# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from helper import plot_permute_importance, plot_feature_importance

%matplotlib inline

# Read the dataset "heart.csv"
df = pd.read_csv("heart.csv")

# Take a quick look at the data 
df.head()

# Assign the predictor and response variables.

# 'AHD' is the response and all the other columns are the predictors
X = df.drop('AHD',axis=1)
y = df['AHD']

# Set the model parameters

# The random state is fized for testing purposes
random_state = 44

# Choose a `max_depth` for your trees 
max_depth = 2

### edTest(test_decision_tree) ###

# Define a Decision Tree classifier with random_state as the above defined variable
# Set the maximum depth to be max_depth
tree = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)

# Fit the model on the entire data
tree.fit(X, y);

# Using Permutation Importance to get the importance of features for the Decision Tree 
# with random_state as the above defined variable
tree_result = permutation_importance(tree, X, y, random_state=random_state)

### edTest(test_random_forest) ###

# Define a Random Forest classifier with random_state as the above defined variable
# Set the maximum depth to be max_depth and use 10 estimators
forest = RandomForestClassifier(random_state=random_state, max_depth=max_depth, n_estimators=10)

# Fit the model on the entire data
forest.fit(X, y);

# Use Permutation Importance to get the importance of features for the Random Forest model 
# with random_state as the above defined variable
forest_result = permutation_importance(forest, X, y, random_state=random_state)

# Helper code to visualize the feature importance using 'MDI'
plot_feature_importance(tree,forest,X,y);

# Helper code to visualize the feature importance using 'permutation feature importance'
plot_permute_importance(tree_result,forest_result,X,y);

