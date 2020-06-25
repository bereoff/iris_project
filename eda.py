import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Data set reading

iris = sns.load_dataset("iris")

## Data set infos.

iris.head(50)

iris.info()

iris.describe()

# the data set deals with three different flower species: 'setosa', 'virginica' and 'versicolor' 
iris['species'].value_counts()

## EDA - to understand and get insights from the data set

# A perception of missing values in this data set

iris.isnull().sum()
# Don't have any missing values

# A view from the data

sns.set_style(style='white')
plt.figure(figsize=(9,7))
sns.boxplot(data=iris)
plt.show()

# Countplot sepal_length

plt.figure(figsize=(10,8))
plt.tight_layout()
sns.countplot(x='sepal_length',data=iris)
plt.show()

# Countplot sepal_width

plt.figure(figsize=(10,8))
plt.tight_layout()
sns.countplot(x='sepal_width',data=iris)
plt.show()

# Countplot petal_length

plt.figure(figsize=(10,8))
plt.tight_layout()
sns.countplot(x='petal_length',data=iris)
plt.show()

# Countplot petal_width

plt.figure(figsize=(10,8))
plt.tight_layout()
sns.countplot(x='petal_width',data=iris)
plt.show()

# Getting a perception of which flower seems to be most separable
sns.pairplot(iris,hue='species',height=1.65)
plt.show()
# Setosa looks like to be most separable among them

# Getting a perception of which feature has a greater correlation among them

plt.figure(figsize=(6,4))
plt.tight_layout()
sns.heatmap(iris.corr(),annot=True,cmap='gist_earth')
plt.show()
# looks like petal_lenght and petal_width have a stronger correlation than other features

## Saving the data set

iris.to_csv(path_or_buf='C:/Users/bbere/data_science_proj/iris_proj/iris_dataset.csv',index=False)



