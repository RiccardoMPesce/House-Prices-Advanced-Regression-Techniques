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
