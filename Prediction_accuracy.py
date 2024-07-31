# In the previous exercise, you created the dataset required to predict who among the infected needs to be given a hospital bed immediately. The aim of this exercise is to use that train and test data to train a simple kNN model.

# Import necessary libraries

# Your code here
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Read the datafile "covid_train.csv"
df_train = pd.read_csv("covid_train.csv")

# Take a quick look at the dataframe
df_train.head()

# Read the datafile "covid_test.csv"
df_test = pd.read_csv("covid_test.csv")

# Take a quick look at the dataframe
df_test.head()

# Get the train predictors
X_train = df_train.drop(['Urgency'],axis=1)

# Get the train response variable
y_train = df_train['Urgency']

# Get the test predictors
X_test = df_test.drop(['Urgency'],axis=1)

# Get the test response variable
y_test = df_test['Urgency']

# Define the range of k values to test
k_values = range(1, 21)  # Test k from 1 to 20
mean_scores = []

# Set up KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)  # 5-fold cross-validation

# Evaluate k-NN for each k
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')  # Use accuracy as the scoring metric
    mean_scores.append(scores.mean())

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_scores, marker='o')
plt.xlabel('Number of Neighbors k')
plt.ylabel('Mean Cross-Validation Accuracy')
plt.title('k-NN: Accuracy vs. k')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Find the best k value
best_k = k_values[np.argmax(mean_scores)]
print(f'The optimal k value is {best_k} with a cross-validation accuracy of {max(mean_scores):.2f}')

### edTest(test_model) ###

# Define your classification model
model = KNeighborsClassifier(n_neighbors=best_k)

# Fit the model on the train data
model.fit(X_train,y_train)

### edTest(test_accuracy) ###

# Predict on the test data
y_pred = model.predict(X_test)

# Compute the accuracy on the test data
model_accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy is {model_accuracy}")
