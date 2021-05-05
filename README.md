# Classify iris plants into three species: Project Overview
* Created a model based on Machine Learning to predict which species is among 3 specific iris species.
* Comparison among Machine Learning models, for an assertive result analysis.

# Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn
**
 
## EDA
I looked at the correlation of the data to identify possible patterns. Below are a few highlights. 

![alt text](https://github.com/Bereoff/iris_project/blob/master/_pairplot_.png "pairplot of the data")
![alt text](https://github.com/Bereoff/iris_project/blob/master/_heatmap_.png "correlation among features")

## Model Building 
First, I splitted the data into train and test sets with a test size of 30%.   

I tried three different models and evaluated them:

*	**SVC** – Baseline for the model
*	**Gridsearch** – I tried to enhance the performance of the model, comparatively to SVC
*	**Random Forest** – Comparison with the baseline model. 

## Model performance
The SVC/GridSearchCV models had a better performance than Random Forest model, with higher accuracy, but all models worked well,demonstrating an assertive chose about the prediction models.Below are the results from the evaluated models. 


![alt text](https://github.com/Bereoff/iris_project/blob/master/confusion_matrix_svc.png "SVCconfusion matrix")
![alt text](https://github.com/Bereoff/iris_project/blob/master/confusion_matrix_grid.png "GridSearchCV confusion matrix")
![alt text](https://github.com/Bereoff/iris_project/blob/master/confusion_matrix_rfc.png "Random Forest confusion matrix")
