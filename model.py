#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Predict the salary(depandent variable) from Years of experience(Indepedent variable.) 
#Just apply linear regression algroithm use the data set test,train. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#Import dataset from directory
path=r'salary_data.csv'
ps = pd.read_csv(path)

#Independent and dependent variables
year_exp = ps.iloc[:, :-1].values #get a copy of dataset exclude last column
sal = ps.iloc[:, 1].values #get array of dataset in column 1st

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(year_exp, sal, test_size=1/3, random_state=0) #70%,30%


# Fit the single linear regression
reg = LinearRegression()
reg.fit(X_train, y_train)
print(reg.intercept_, reg.coef_[0])
print(reg.score(X_train, y_train))

# Saving model to disk
pickle.dump(reg, open('model.pkl','wb'))

# Loading model to compare the results
# Predict the result of salary based on Years of experience
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2]]))
