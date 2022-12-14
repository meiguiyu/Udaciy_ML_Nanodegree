# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This dataset contains data about candidates who apply for a banking service. 
We seek to predict if an apply will be approved or not based on the features collected for this candidate.

The best performing model was a VotingEbsemble model from AutoML.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
In this pipeline, I am going to compare the performace of Logistic Regression model and AutoML model in predicting if a banking application will be approved or not.
First the data is loaded using TabularDatasetFactory class. Then C and max_iter hyperparameters are tuned in the logistic regression. Early stopping policy is applied to control overfitting.
And for the AutoML, cross validation is enabled. Accuracy is the primary metric in both models.

**What are the benefits of the parameter sampler you chose?**
The parameter C can help prevent over fitting and max_iter can optimize the computation time. 
Have these two parameter tuned, the algorithm is able to find a good result quickly without overfitting.

**What are the benefits of the early stopping policy you chose?**
With the evaluation_interval set to 5 and slack_factor set to 0.1, I want evaluate the primary metric every 5 iteration and if the value falls outside top 10% of the primary metric then 
the training process will stop. This can avoid over fitting the noise in the training data.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
VotingEnsemble was selected as the best AutoML model with an accuracy as 0.917. 
The best hyperparameters are "colsample_bytree" = 0.7, eta=0.1, gamma = 0.1, max_depth = 9, max_leaves = 511, n_estimators = 25.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
The Logistic Regression model's accuracy is 0.912 and the AutoML model's accuracy is 0.917. Since this is not an imbalaced dataset and the accuracy score works well to identify the best
model. So based on the accuracy socre, I would say both model are performing well and AutoMl is a very slightly better model.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Derive new variables from existing dataset to improve accuracy. Or add new features into the dataset.

## Delete computer cluster
![alt text](Capture.PNG)


