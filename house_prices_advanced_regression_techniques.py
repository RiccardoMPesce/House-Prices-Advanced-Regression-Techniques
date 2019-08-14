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

# Mutating columns
train_df = train_df.assign(TotalSF=train_df["1stFlrSF"]+train_df["2ndFlrSF"])
test_df = test_df.assign(TotalSF=test_df["1stFlrSF"]+test_df["2ndFlrSF"])

# Dropping columns
train_df.drop(cols_to_drop, axis=1, inplace=True)
test_df.drop(cols_to_drop, axis=1, inplace=True)

# Selecting numerical and categorical columns
cat_cols = train_df.select_dtypes(exclude="number").columns.values.tolist()
num_cols = train_df.select_dtypes("number").columns.values.tolist()
# Dropping the response variable from the selection of numerical columns
num_cols.remove("SalePrice")

#%% [markdown]
# ### Imputing NA values
# Before encoding, we have to impute NA values. Having dropped the columns whose values are mostly NAs, we can now use strategies to impute. We will impute categorical NAs with the class "None" to represent that, for that variable, that feature is not present. For the continuous and ordinal features corresponding or relating to a None, we will set the value of 0.

#%%
# Imputing categorical columns
train_df[cat_cols] = train_df[cat_cols].fillna("None")
test_df[cat_cols] = test_df[cat_cols].fillna("None")

# Imputing numerical columns
train_df[num_cols] = train_df[num_cols].fillna(0)
test_df[num_cols] = test_df[num_cols].fillna(0)

# Displaying data
with pd.option_context("display.max_columns", None):
    display(train_df.head(20))

#%% [markdown]
# ### Encoding categorical variables
# Now that we have imputed NA values, we need to encode our categorical variables so as to fit the model.
#
# Some variables are binary (Yes/No), some others are ordinal (they present an intrinsic order) while some others don't have an order: for the first ones, we define some dicts and map these features in this uniform way. For the latter, we use OneHotEncoding. First we define each list of columns with the name refferring to the mapping we want to use. Then, we apply that mapping with `pandas` built-in method `map()`.

#%%
street_air_cols = ["Street", "PavedDrive", "CentralAir"]
shape_cols = ["LotShape"]
slope_cols = ["LandSlope"]
utility_cols = ["Utilities"]
flatness_cols = ["LandContour"]
qual_cond_exp_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish"]

qual_cond_exp_dict = {
    "No": 0, "None": 0, 
    "Unf": 0.5,
    "RFn": 0.7,
    "Po": 1, "LwQ": 1, "Mn": 1, "Fin": 1,
    "Fa": 2, "Rec": 2, "Av": 2,
    "TA": 3, "BLQ": 3,
    "Gd": 4, "ALQ": 4,
    "Ex": 5, "GLQ": 5
}

street_air_dict = {
    "Gravel": 0, "N": 0,
    "P": 0.5,
    "Paved": 1, "Y": 1
}

shape_dict = {
    "Reg": 0,
    "IR1": 1,
    "IR2": 2,
    "IR3": 3
}

flatness_dict = {
    "Low": -1,
    "Lvl": 0,	
    "Bnk": 1,
    "HLS": 2
}

utility_dict = {
    "ELO": 0,
    "NoSeWa": 1,
    "NoSewr": 2,
    "AllPub": 3	
}

slope_dict = {
    "Gtl": 0,
    "Mod": 1,	
    "Sev": 2
}

# Encoding ordinal columns
train_df[street_air_cols] = train_df[street_air_cols].replace(street_air_dict)
test_df[street_air_cols] = test_df[street_air_cols].replace(street_air_dict)

train_df[shape_cols] = train_df[shape_cols].replace(shape_dict)
test_df[shape_cols] = test_df[shape_cols].replace(shape_dict)

train_df[utility_cols] = train_df[utility_cols].replace(utility_dict)
test_df[utility_cols] = test_df[utility_cols].replace(utility_dict)

train_df[flatness_cols] = train_df[flatness_cols].replace(flatness_dict)
test_df[flatness_cols] = test_df[flatness_cols].replace(flatness_dict)

train_df[qual_cond_exp_cols] = train_df[qual_cond_exp_cols].replace(qual_cond_exp_dict)
test_df[qual_cond_exp_cols] = test_df[qual_cond_exp_cols].replace(qual_cond_exp_dict)

with pd.option_context("display.max_columns", None):
    display(train_df.head(20))

#%% [markdown]
# Now that all the ordinal categorical variables have been converted to numbers, we need to fit a One-Hot encoder to the remaining categorical variables, so that we can analyze the correlation.
#
# We could have done more feature engineering if we had more data (we could transform other variables such as "Steet" in a way that it would become categorical).
#
# Let's fit a One-Hot Encoder, but first let's select the remaining categorical columns.

#%%
left_cats = ["MSZoning", "Street", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "Electrical", "Functional", "GarageType", "SaleType", "SaleCondition"]

#%% [mardown]
# Before proceeding, we need to stack both training and test set, and apply our encoder to the new temporary, set. In this way, we will avoid to find unknown categories during testing.

#%%
X = train_df.drop("SalePrice", axis=1)
y = train_df["SalePrice"]

# To appropriately split back
train_size = train_df.shape[0]

temp_df = pd.concat([X, test_df])

# Getting idicator
temp_df = pd.get_dummies(temp_df)

# Splitting back train and test sets
X = temp_df.iloc[:train_size, :]
test_df = temp_df.iloc[train_size:, :]

train_df = pd.concat([X, y], axis=1)

with pd.option_context("display.max_columns", None):
    display(train_df.head(20))

#%% [markdown]
# ### Correlation among variables
# We calculate the correlation matrix of the features, so as to see which ones are most related with the response (both positively and negatively). 
# In particular, `corr_mat` contains the correlation matrix while `corr_y` contains the individual correlations of each predictor to the response. 
# We will remove those columns whose correlation to the response is less, in absolute v

#%%
corr_mat = train_df[num_cols + ["SalePrice"]].corr()
corr_threshold = 0.5

# Correlation with the response
corr_y = corr_mat["SalePrice"]
low_corr_cols = corr_y[np.abs(corr_y) < 0.5]
# Converting to list
low_corr_cols = low_corr_cols.index.values.tolist()

with pd.option_context("display.max_rows", None):
    print(corr_y)

# Printing the number of columns and the names we will drop
print("The number of weakly correlated variables to the response (with correlation threshold set at " + str(corr_threshold) + ") is " + str(len(low_corr_cols)) + ".")

# Printing such columns
print(low_corr_cols)

# Printing number of total columns (except response)
print(len(corr_y) - 1)

#%% [markdown]
# So, with 225 - 34 = 191 columns, we have reduced our dataset size.
#
# Let's now remove these columns, and finally split our dataset into X an y.

#%%
train_df = train_df.drop(low_corr_cols, axis=1)
test_df = test_df.drop(low_corr_cols, axis=1)

for e in low_corr_cols:
    num_cols.remove(e)

# Displaying data
with pd.option_context("display.max_columns", None):
    display(train_df.head(20))

X = train_df.drop("SalePrice", axis=1)
y = train_df["SalePrice"]

#%% [markdown]
# Let's now plot a heatmap with our correlation matrix. Its purpose is to visualize highly correlated variables.
#
# Again, if two variables are highly correlated, we will keep the one who is best correlated with the response

#%%
import seaborn as sns
import matplotlib.pyplot as plt

print(train_df[num_cols + ["SalePrice"]].columns)

plt.figure(figsize=(15, 15))
hm = sns.heatmap(train_df[num_cols + ["SalePrice"]].corr(), annot=True, center=0, square=True)
# Let's save the plot as a figure
hm.get_figure().savefig("corr.png")
plt.show()

#%% [markdown]
# Here we see that several variables are highly correlated one another. TotalSF and GrLivArea vary identically. We remove the latter since it less slightly less correlated to the response.
# 
# Using this approach, let's get rid of some columns.

#%%
cols_to_remove = ["GrLivArea", "TotRmsAbvGrd", "GarageArea"]

train_df = train_df.drop(cols_to_remove, axis=1)
test_df = test_df.drop(cols_to_remove, axis=1)
X = X.drop(cols_to_remove, axis=1)


#%% [markdown]
# ### Random Forest
# Now, we can fit our random forest model. First we divide X, y in train and validation, and we will see their performances.

#%%
import csv

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rf = RandomForestRegressor(n_jobs=-1)

rf.fit(X_train, y_train)

preds = rf.predict(test_df)

id = 1461
with open("submission.csv", "w") as f:
    f.write("Id,SalePrice\n")

with open("submission.csv", "a") as f:
    for n in preds:
        f.write(str(id) + "," + str(n) + "\n")
        id += 1


#%% [markdown]
# In Kaggle, we got a Root Mean Squared Logarithmic Error of 0.15575. It can be improved. We'll come back at this later.
