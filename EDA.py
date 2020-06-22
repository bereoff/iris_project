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
# Don´t have any missing values

# Getting a perception of which flower seems to be most separable

sns.set_style(style='darkgrid')
sns.pairplot(iris,hue='species')
plt.show()
# Setosa looks like to be most separable among them

# Getting a percepetion of which feature has a greater correlation among them

sns.heatmap(iris.corr(),annot=True,cmap='viridis',cbar_kws={'shrink':0.5})
plt.show()
# looks like petal_lenght and petal_width have a stronger correlation than other features --> # Plot petal_length x Petal_width

sns.jointplot(x='petal_width',y='petal_length',data=iris,kind='reg')
plt.show()





