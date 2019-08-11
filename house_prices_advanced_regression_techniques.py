# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
# ## House Prices: Advanced Regression Techniques
# ### Competition Description: 
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.
# 
# #### Let's get started
# We will first import the necessary libraries.

#%%
import numpy as np
import pandas as pd

import os

#%% [markdown]
# I will define here some constants regarding the path of our files and the names of the folders. To guarantee that our program runs on every operating system, we will use `os.path.join()` function (from the `os` library) to create system-independent path to our .csv files.

#%%
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
FOLDER_NAME = "data"

TRAIN_PATH = os.path.join(FOLDER_NAME, TRAIN_FILE)
TEST_PATH = os.path.join(FOLDER_NAME, TEST_FILE)

#%% [markdown]
# #### Importing the data
# Let's import the data using panda's `read_csv()` function.

#%%
train_df = pd.read_csv(TRAIN_PATH)

# test_df will only be used to create the submission .csv file
test_df = pd.read_csv(TEST_PATH)

#%% [markdown]
# #### Visualizing the data
# Next step is to visualize the data (the first 50 observations) in order to try and understand the predictors, and what kind of relationship there can be between them and the response.
# This step is necessary before data preprocessing.

#%%
print("Shape of train_df: " + str(train_df.shape))
print("Shape of test_df: " + str(test_df.shape))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(train_df.head(50))
    display(test_df.head(50))

#%% [markdown]
# We can see that there are different NaN values that should be taken care of before proceeding further. Our strategy is either to impute (i.e. substitute) the NaN value with the mean of the relative column, or, if the vast majority of that column's values are NaN, we simply drop it.
# Let's find the columns whose 50% or more values are missing. To do so, we will print the percentage of missing values for each column, with the options to display every row of the obtained panda series.

#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(train_df.isna().mean().round(2))

#%% [markdown]
# We see that PoolQC has every value missing, so we wil drop it. We will also drop MiscFeature as 96% of values are missing, Fence which has 81% of values missing and Alley with 94% of values missing.

#%%
cols_to_drop = ["PoolQC", "MiscFeature", "Fence", "Alley"]

X = train_df.drop("SalePrice", axis = 1)
y = train_df["SalePrice"]

X = X.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)

#%% [markdown]
# #### Categorical variables imputation
# We now impute categorical variables using the constant value strategy, that is we replace the NAs with the string "Unknown". We will also convert such columns into categorical ones "forcibly".

#%%
# Selecting categorical columns
cat_cols = list(X.select_dtypes(exclude="number").columns)

X[cat_cols] = X[cat_cols].fillna("Unknown")
X[cat_cols] = X[cat_cols].astype("category")
test_df[cat_cols] = test_df[cat_cols].fillna("Unknown")
test_df[cat_cols] = test_df[cat_cols].astype("category")

# Let's peek at our new dataset
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(X.head(n=100))

#%% [markdown]
# #### Numerical variables imputation
# As with our categorical variables, we need to impute our numerical NaNs too. To do so, we impute them with the value 0.

#%%
# Selecting numerical columns
num_cols = list(X.select_dtypes(include="number").columns)

X[num_cols] = X[num_cols].fillna(0)
test_df[num_cols] = test_df[num_cols].fillna(0)

# Let's peek at our new dataset
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(X.head(n=100))

#%% [markdown]
# #### Encoding categorical variables
# In order to encode categorical variables, we will use an ordinal encoder for those features whose encoding represents an ordering, while we will use one-hot encoding to encode categorical features whose values don't follow any order. 
#
# In our dataset, no categorical variable has an order, therefore we have to use one-hot encoding for any categorical.

#%%
from category_encoders.one_hot import OneHotEncoder
enc = OneHotEncoder(use_cat_names=True)

enc.fit(X, y)

X = enc.transform(X)
test_df = enc.transform(test_df)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(X.head(n=100))

#%% [markdown]
# #### Fitting our first model (Random Forest)
# The next step will be to split our dataset into train and validation set. Validation set will help us predict test accuracy. Since we are dealing with temporal data, we are going to make sure that our validation set will be the newest sold records (2009-2010), while the previous one will make the training set.
# Once we made this split, we will fit our model with different numbers of estimators, and we will pick the one with the highest score (R<sup>2</sup>).

#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Selecting indices of rows
i_train = list(X["YrSold"] < 2009)
i_test = list(X["YrSold"] >= 2009)

X_train, y_train = X[i_train], y[i_train]
X_test, y_test = X[i_test], y[i_test]

print(len(y_train))
print(len(y_test))

# Using different numbers of estimators


#%%



