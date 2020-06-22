import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

## Data set reading

iris = sns.load_dataset("iris")

## Data set info

iris.head(50)

## Splitting the data into a training set and testing set

X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train a model

model = SVC()

model.fit(X_train,y_train)

# Model Evaluation

predictions = model.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


