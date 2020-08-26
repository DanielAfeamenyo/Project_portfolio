#!/usr/bin/env python
# coding: utf-8

# # BOSTON HOUSING DATASET FOR REGRESSION MACHINE LEARNING¶
# 

# # Import libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Load Boston Housing Dataset from the scikit learn repository¶
# 

# In[3]:


from sklearn.datasets import load_boston
boston = load_boston() 


# # Summarize the Dataset
#    Taking a look at the data in the following ways:
# 
#    Peek at the data Quality check Statistical summary of all attributes
# 
# 

# # Peek at the data

# In[4]:


# Print out a description of the dataset.
print(boston.DESCR)


# In[5]:


# Take a look at the column names.
boston.feature_names


# In[6]:


# Convert the dataset into a pandas dataframe.
df = pd.DataFrame(boston.data, columns = boston.feature_names)


# In[7]:


# Take a look at the first 5 rows of the data.
df.head()


# House prices (MEDV) did not appear in the dataframe. They are the target of the boston dataframe and therefore, needs to be added to the dataframe.

# In[8]:


df['MEDV'] = boston.target


# In[9]:


df.head()


# # Quality Check
#    This is to check if there are null-values in the dataset.

# In[10]:


df.info()


# # Statistical summary of all attributes

# In[11]:


df.describe()


# # Visualize the Data
#    Making a quick visualization for the data. This is to:
# 
# Understand the distribution of each feature Find the correlation between the features Identify the features that correlates most with the House price(MEDV

# In[12]:


sns.pairplot(df);


# # Distribution

# In[18]:


rows = 7
cols = 2
fig, ax = plt.subplots(nrows= rows, ncols= cols, figsize = (16,16))
col = df.columns
index = 0
for i in range(rows):
    for j in range(cols):
        sns.distplot(df[col[index]], ax = ax[i][j], kde_kws = {'bw' : 1})
        index = index + 1
plt.tight_layout()


# # Correlation
# Summarize the relationships between the variables

# In[23]:


fig, ax = plt.subplots(figsize = (16,9))
sns.heatmap(df.corr(), annot=True, annot_kws={'size':12});


# From the correlation matrix, in regard to our target column, it is observed that the house pricing(MEDV) has a strong positive relationship with RM and strong negative relationship with LSTAT.
# 
# -It also has moderate positve/negative relationship with the other columns.
# 
# -For a linear regression method, a nearly high correlation is needed hence a threshold filter must be defined.
# 
# -For this reason, we need to define a function called getCorrelatedFeature

# In[24]:


def getCorrelatedFeature(corrdata, threshold):
    feature = []
    value = []
    
    for i, index in enumerate(corrdata.index):
        if abs(corrdata[index])> threshold:
            feature.append(index)
            value.append(corrdata[index])
    
    df = pd.DataFrame(data = value, index = feature, columns = ['Corr Value'])
    return df


# In[25]:


# Setting a threshold limit of 0.4.
threshold = 0.4
corr_value = getCorrelatedFeature(df.corr()['MEDV'], threshold)


# In[26]:


# Checking out the dependencies after applying the threshold limit
corr_value.index.values


# # Quick view of correlated data

# In[27]:


correlated_data = df[corr_value.index]
correlated_data.head()


# # Linear Regression
# 
# 

# # Split and Test dataset

# In[28]:


X = correlated_data.drop(labels=['MEDV'], axis = 1)
y = correlated_data['MEDV']


# In[32]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state=1)


# In[33]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[34]:


lm.fit(X_train,y_train)


# # Results visualization

# The goal is to have a perfect linear or nearly linear relation of the points. The larger the distribution of the points, the greater the inaccuracy of the model

# In[35]:


# predict with the X_test data
predictions = lm.predict(X_test)


# In[37]:


# plot the y_test against the predicted values on a scatter plot
plt.scatter(y_test,predictions);


# In[39]:


sns.distplot((y_test-predictions),bins = 50);


# # Linear regression function¶

# In[40]:


# Define linear regression function.
def lin_func(values, coefficients=lm.coef_,y_axis=lm.intercept_):
    return np.dot(values, coefficients) + y_axis


# # Prediction Samples

# In[41]:


from random import randint
for i in range(5):
    index = randint(0, len(df)-1)
    sample = df.iloc[index][corr_value.index.values].drop('MEDV')
    print('PREDICTION: ', round(lin_func(sample),2),
         '// REAL: ', df.iloc[index]['MEDV'],
         '// DIFFERENCE: ', round(round(lin_func(sample),2)-df.iloc[index]['MEDV'],2))


# # The dependencies in the model are:

# INDUS, NOX, RM, TAX, PTRATIO and LSTAT
# 
# Because they have the |correlation coefficients|>0.4 which is the threshold value.

#                         By DANIEL AFEAMENYO

# In[ ]:




