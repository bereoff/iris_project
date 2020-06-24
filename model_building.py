import pandas as pd
import numpy as np
import seaborn as sns

## Data set reading

iris = sns.load_dataset("iris")

## Data set info

iris.head()

## Splitting the data into a training set and testing set
from sklearn.model_selection import train_test_split

X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# SVM
from sklearn.svm import SVC

# Model training
model = SVC()

model.fit(X_train,y_train)

# Model Prediction & Evaluation
from sklearn.metrics import confusion_matrix, classification_report

predictions = model.predict(X_test)

#Results

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Model training
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

# Model Prediction & Evaluation

dtree_pred = dtree.predict(X_test)

#Results

print(confusion_matrix(y_test,dtree_pred))
print(classification_report(y_test,dtree_pred))

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Model training
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

# Model Prediction & Evaluation

rfc_pred = rfc.predict(X_test)

#Results

print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

