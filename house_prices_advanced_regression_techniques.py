#%% [markdown]
# # House Prices: Advanced Regression Techniques
# ### [Link to the challenge on Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/)
# ## Challenge Description
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
#
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

#%% [markdown]
# ### Objective
# The objective of the challenge is to build a regression model that predicts the sale price of an house given its features, which are the 79 explanatory variables.
#
# Before fitting a regression model, we need to understand the data in order to spot the features that matter the most and in order to drop those features that aren't relevant to the response, or are highly correlated with other predictors (so we avoid multicollinearity).
#
# Let's get started by importing our data first, after defining our files' paths.

#%%
import numpy as np
import pandas as pd

# We need os.join.path() to create system-independent paths
import os

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
DATA_FOLDER = "data"

train_path = os.path.join(DATA_FOLDER, TRAIN_FILE)
test_path = os.path.join(DATA_FOLDER, TEST_FILE)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

#%% [markdown]
# Now that we have imported our data, let's give it a look, by displaying the first 20 rows. We will print each column so as to get a general idea of the predictors.

#%%
with pd.option_context("display.max_columns", None):
    display(train_df.head(20))

#%% [markdown]
# ### Feature Engineering 
# We can see that there are some columns whose values are mostly NAs. Since they won't bring any improvement to our models, there can be dropped from the dataset. Let's first see the fraction of NAs for each column.

#%%
with pd.option_context("display.max_rows", None):
    print(train_df.isnull().mean().round(2))

#%% [markdown]
# We can see that MiscFeature, Fence, PoolQC, Alley are mainly composed of NAs. We will drop them. 
#
# The explanatory variable "FireplaceQu" has NaN whenever the variable "Fireplaces" is zero, i.e. absent. We will then set it to "Absent" where it has NA. This variable is an ordinal categorical variable (since it deals with quality). There are several categorical variables that have an intrinsic order, and for them we are going to use an ordinal encoder, where the absence will be indicated with 0.
#
# Some variables are highly correlated one another. We will see them with the correlation heatmap plotted in SeaBorn. For now, we can see that TotalBsmtSF is just the sum of BsmtFinSF1 and BsmtFinSF2, therefore the latter two can be dropped. Since the sale price will depend on both "1srFlrSF" and "2ndFlrSF" in conjunction, we can combine these two columns into one called "TotalSF".
#
# Binary columns indicate that such variable can hold a binary value (yes/no) and will be encoded as such.

#%%
cols_to_drop = ["MiscFeature", "Fence", "PoolQC", "Alley", "BsmtFinSF1", "BsmtFinSF2", "1stFlrSF", "2ndFlrSF"]
ordinal_cols = ["ExterQual", "ExterCond", "LandSlope", "BsmtQual", "BsmtCond", "BsmtExposure", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
binary_cols = ["CentralAir", "PavedDrive"]

# Selecting numerical and categorical columns
cat_cols = train_df.select_dtypes(exclude="number")
num_cols = train_df.select_dtypes("number")

# Mutating columns
train_df = train_df.assign(TotalSF=train_df["1stFlrSF"]+train_df["2ndFlrSF"])
test_df = test_df.assign(TotalSF=test_df["1stFlrSF"]+test_df["2ndFlrSF"])

# Dropping columns
train_df.drop(cols_to_drop, axis=1, inplace=True)
test_df.drop(cols_to_drop, axis=1, inplace=True)

#%%
