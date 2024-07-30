# In this exercise we are interested in fitting a linear regression model to predict units sold (sales) from TV advertising budget (tv). But when fitting a linear regression, slight variations in the training data can affect the coefficients of the model.
# To make this issue more concrete, you've been provided with a function RandomUniverse(DataFrame) -> DataFrame that takes a dataset as input and returns a new, slightly different dataset from a "parallel universe." 
# We can fit a regression model to this new, "parallel universe" data to calculate a  β0 ​and 𝛽1 coefficient. This process can be repeated many times, first generating the new dataset with RandomUniverse, and then calculating a new 𝛽0 and 𝛽1 from the new dataset. The resulting collection of 𝛽0 s and 𝛽1s can be plotted as histograms like those below. 

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from randomuniverse import RandomUniverse
%matplotlib inline

# Read the advertising dataset as a pandas dataframe
df = pd.read_csv('Advertising_adj.csv')

# Take a quick look at the dataframe
df.head()

# Create two empty lists that will store the beta values
beta0_list, beta1_list = [],[]

# Choose the number of "parallel" Universes to generate 
# that many new versions of the dataset
parallelUniverses = 100

# Loop over the maximum number of parallel Universes
for i in range(parallelUniverses):

    # Call the RandomUniverse helper function with the dataframe
    # read from the data file
    df_new = RandomUniverse(df)

    # Find the mean of the predictor values i.e. tv
    xmean = df_new['tv'].mean()

    # Find the mean of the response values i.e. sales
    ymean = df_new['sales'].mean()

    # Compute the covariance of X and Y
    covariance = np.cov(df_new['tv'], df_new['sales'])[0, 1]

    # Compute the variance of X
    variance = np.var(df_new['tv'], ddof=0)

    # Compute the analytical values of beta1 and beta0 using the equations
    beta1 = covariance / variance
    beta0 = ymean - beta1 * xmean

    # Append the calculated values of beta1 and beta0 to the appropriate lists
    beta0_list.append(beta0)
    beta1_list.append(beta1)

### edTest(test_beta) ###

# Compute the mean of the beta values
beta0_mean = np.mean(beta0_list)
beta1_mean = np.mean(beta1_list)

# Plot histograms of beta_0 and beta_1 using lists created above 
fig, ax = plt.subplots(1,2, figsize=(18,8))
ax[0].hist(beta0_list, bins=20, edgecolor='k', alpha=0.7)
ax[1].hist(beta1_list, bins=20, edgecolor='k', alpha=0.7)
ax[0].set_xlabel('Beta 0')
ax[1].set_xlabel('Beta 1')
ax[0].set_ylabel('Frequency');
