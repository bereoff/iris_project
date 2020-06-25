import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Data set reading

iris = sns.load_dataset("iris")

## Data set info

iris.head()

## Splitting the data into a training set and testing set
from sklearn.model_selection import train_test_split

X = iris.drop('species',axis=1)
y,levels = pd.factorize(iris['species'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# SVC
from sklearn.svm import SVC

# Model training
model = SVC()

model.fit(X_train,y_train)

# Model Prediction & Evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

predictions = model.predict(X_test)

#Results

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# GridSearch
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}

# Create a GridSearchCV object and fit it to the training data.
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

grid.fit(X_train,y_train)

# Model Prediction & Evaluation

grid_predictions = grid.predict(X_test)

#Results

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))

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


# Confusion Matrix plot

## SVC

cf_svc = confusion_matrix(y_test,predictions)
print(cf_svc)

sns.heatmap(cf_svc,annot=True,cmap='Blue')
plt.show()

# Confusion Matrix heatmap comparison among the models

# SVC

cf_svc = confusion_matrix(y_test,predictions)

hm_svc = pd.DataFrame(cf_svc,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

plt.figure(figsize=(5.5,4))
sns.heatmap(hm_svc, annot=True,cmap='YlGnBu')
plt.title('SVC \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, predictions)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# GridSearchCV

cf_grid = confusion_matrix(y_test,grid_predictions)

hm_grid = pd.DataFrame(cf_grid,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

plt.figure(figsize=(5.5,4))
sns.heatmap(hm_grid, annot=True,cmap='YlGnBu')
plt.title('GridSearchCV \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, grid_predictions)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Random Forest

cf_rfc = confusion_matrix(y_test,rfc_pred)

hm_rfc = pd.DataFrame(cf_rfc,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

plt.figure(figsize=(5.5,4))
sns.heatmap(hm_rfc, annot=True,cmap='YlGnBu')
plt.title('Random Forest \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, rfc_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

