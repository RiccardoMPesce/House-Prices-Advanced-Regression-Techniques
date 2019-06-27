#%% [markdown]
# # House Prices: Advanced Regression Techniques
# #### [Link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) to the relative Kaggle Competition

#%% [markdown]
# ## Brief description
# With Regression we seek to estimate the house price, given 79 predictors (explanatory variables).
# The explanatory variables wholly describe every single aspect of the house, so not every one is 
# needed for our regression.

#%% 
# Importing numpy, pandas and pyplot
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#%% 
# Importing os for handling files and directories in an uniform way across different OS's
import os

#%% 
# Reading the training csv data file into panda dataframe 
training_set = pd.read_csv("data/train.csv")
