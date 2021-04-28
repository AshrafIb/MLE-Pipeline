# Optimizing an ML Pipeline in Azure

Author: Ashraf Ibrahim <br>
Date: 27.04.2021 <br>

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

This Project uses Data from a marketing Campaing of a Banking Company, wherein it was collected whether customers subscribed to a term deposit or not. For each customer, several features where gathered. The predictive purpose of this Analysis is to understand, wehter a Subscription can be predicted or not and of course under consideration of the quality of a prediction. 

Azure Machine-Learning Studio is used as main tool, where two approaches are compared: 
+ Python SDK with Hyperdrive 
+ AutoML 

The best Performace using Hyperdrive was achieved with an Accuracy of 0.911 (using a Logistic Regression), in contrast using AutoML the best performance was an Accuracy of 0.917 based on VotingEnsemble Algorithm. 

## Scikit-learn Pipeline

My Scikit-Learn pipeline consists of the following steps:
+ Download and Preprocessing (Clean and split Data into train and test set)
+ Training a logistic Regression with with a RandomParameterSampling, sampling randomly from
    +  C (the inverse of regularization strength) 
    +  max_iter (the maximum of iteration befor converging)
    +  penalty (the type of regularization - l1 or l2)
+  Saving the best Model and the best Params

A RandomParameterSampling randomly selects Parameters from a specified Range. It is equivalent to Sklearns RandomizedCV, where an Algorithms is used with a Random set of predefined Parameters. This is not as time consuming and exhaustive as a Grid-Sweep, where each combination is tested. 

The RandomParameterSampling did consider a BanditPolicy and thus terminating each run early, which is deviated (by a slack_factor of 0.1) from the best model. This saves computational time and thus ressources. 

The best Model has a C-value of 1.7, a Max-Iteration of 89 and a l1-Penalty.

## AutoML

The AutoML-Variant followd the same steps:

+ Download and Preprocess the Data
+ Configuration of AutoML 
    ```automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task='classification',
    compute_target=cluster,
    primary_metric='accuracy',
    training_data=dataset,
    label_column_name='y',
    n_cross_validations=5,
    max_concurrent_iterations=4,
    max_cores_per_iteration=-1)```
    ```
+ Submitting the Run  

Finally, AutoML identifies a VotingEnsemble as the best Classifier for this Task. A VotingEnsemble is a Meta-Classifier, consisting of several Classifiers, where the final Classification is based on a majority voting. This VotingEnsemble consists of 36 different Classifiers based on three Algorithms: XGBoostClassifier, LightGBMClassifier and LogisticRegression. Each Classifier is equipped with different Parameters. For Example on Version of a XGBoost is build on the following params:

Example 1:
```python 
XGBoostClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.5, eta=0.1,gamma=0,learning_rate=0.1, max_delta_step=0, max_depth=6, max_leaves=15, min_child_weight=1, missing=nan, n_estimators=100, n_jobs=-1, nthread=None, objective='reg:logistic', random_state=0, reg_alpha=0, reg_lambda=2.0833333333333335, scale_pos_weight=1, seed=None, silent=None, subsample=1, tree_method='auto',verbose=-10,verbosity=0)
```

Example 2: 
```python
XGBoostClassifier(base_score=0.5, booster='gbtree',colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.7, eta=0.05, gamma=1,grow_policy='lossguide', learning_rate=0.1, max_bin=63, max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1, missing=nan, n_estimators=100, n_jobs=-1,  nthread=None, objective='reg:logistic',
random_state=0, reg_alpha=1.5625, reg_lambda=1.0416666666666667, scale_pos_weight=1, seed=None, silent=None,subsample=0.7, tree_method='hist',
verbose=-10, verbosity=0)
```

Both approaches differ marginally. For a detailed view, take a look into my JupyterNotebook where the Parameters of all 36 Algorithms are printed. 

The Best Version used a VotingEnsemble and did perform slightly better,than the Logistic Regression. The Voting Classifier achieves an Accuracy of 0.917. 


## Pipeline comparison

AutoML is easier to use, when several Algorithms should be compared, because it takes fewer steps to configure the Grid of Comparision. In my Hyperdrive approach, i have to make a decision for a specific Algorithm at first and then have to define the Params i want to iterate over. AutoML does this automatically and chooses from all Classification-Algorithms and picks randomly Paremeters. Thus AutoML is a faster possibility to train models. 

Even if the accuracy is slightly better in the AutoML variant, I would not consider this because the data is unbalanced. Therefore, other metrics are more important; I will discuss this in the next section. 


## Future work

In a next run, i would use an AutoML approach again, but focusing on a F1-Score, or in discussion with a business expert try to understand the importance of precision and recall in this context and determine an optimal curve of both. 

I would than take the best performing Algorithm and define a Hyperdrive for it, where i go into a mirco-analysis of best performing Parameters. 

## Proof of cluster clean up

The Cluster was deleted via code in my Notebook. 