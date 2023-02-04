# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:02:58 2022
@author: Abhilash Adya, Belaul Emon Hasan

This code provides visual representation of the data in raw form (before scaling operation was done).

Note: 1) Please assign the path where you have stored the dataset(.csv file) in 'path' variable below.
      2) One can see the graph of other columns like 'new_cases' and 'population' vs year_week just by changing the value 
         in variable 'y_co'.
"""


# importing necessary libraries/functions/classes.

import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

# Assigning Variables

y_co = "new_cases"
x_co = 'year_week'
path = os.path.join(os.getcwd(),"Original_data.csv")
warnings.filterwarnings('ignore')

# New_cases is the new covid cases every week
# Year_weak signifies the year of the data and the number of week. i.e. 2020-W01 means first week of year 2020.

# Reading given data

data = pd.read_csv(path)


# Checking for null datasets

data[y_co].isnull().values.any()


# Removing the null values from data

data = data.dropna(axis=0, subset=[x_co, y_co])
#print(data[y_co].isnull().values.any())


# Using groupby function to sum the data for all the given countries with respect to year_week column.
var = dict(data.groupby(x_co)[y_co].sum())


# Converting obtained dictionary into dataframe. Easy to plot.

df = pd.DataFrame(var.items(), columns=[x_co, y_co])
print(df)

# Plot the dataframe

plt.figure(figsize=(12,6))
ax = df.plot(x= x_co , y= y_co, kind="line")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40)   # rotating x-axis values to avoid overlapping of values
#plt.ticklabel_format(style='plain', axis='y')           # removes scientific notation
plt.title("COVID-19 DATA FOR EUROPE")
plt.ylabel('New Covid cases',fontweight='bold')
plt.xlabel('weekly time data fromm 2020 to 2022',fontweight='bold')
plt.tight_layout()
plt.show()