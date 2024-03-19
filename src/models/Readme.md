# A guide to the ML models's training approach

To select the best performing training run, we log a set of tags, hyperparameters, and metrics. Each of these is listed in detail below.

This document ends with analyses performed to reduce the technical debt of the ML model (read more on this topic here: https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf). 

## Spark ML



## Logging

Currently, we use mlflow to track the training experiments (https://mlflow.org/docs/latest/index.html#). MLflow logs the tags, parameters, metrics, and model artifacts in the Azure Machine Learning workspace.

### Tags

### Hyperparameters

- poly_expansion_degree: the degree to which the pipeline performs the polynomial expansion for feature crossing
- ElasticNetParam: a value that goes from 0 to 1 where a value of 0 specifies only l2 regularization and a value of 1 only l1 or lasso regularization
- RegParam: the parameter that strengthens l2 regularization when it is larger. If both ElasticNetParam and RegParam are 0 then there is no regularization at all
- Standardization: a boolean to specify whether the pipeline standardizes the inputs by subtracting the mean and dividing by the standard deviation prior to inference
- Threshold: chosen to map from flooding probabilities to predictions
- Tol: tells the optimization algorithm to terminate training if improvement between steps is less than the tolerance
- AggregationDepth: from Assaad's understanding, the number of parallel models to train to reduce over-fitting and improve performance



### Metrics

- Test set area under the Receiver Operating Characteristic curve (AUC, https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- Train set AUC: The purpose is to compare train set and test set AUC to identify cases of over-fitting
- Precision & Recall (https://en.wikipedia.org/wiki/Precision_and_recall)
- F1 score: combines precision and recall to get a score that accounts for both
- Area under the Precision-Recall curve
- True positive, true negative, false positive, and false negative rates

Especially for smaller runoff values, the number of flooded pixels is likely exceeded by the number of dry pixels. The precision, recall, and F1 score are solid metrics to evaluate the performance of the data with special attention to flooded pixels. We envision clients to be more satisfied with a model that maximises the F1 score rather than raw accuracy because flooded areas are typically the center of attention for climate mitigation. 

### Model Artifacts

## Interpretability and Explainability

References:
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability-aml
- https://github.com/interpretml/interpret-community/
- https://github.com/microsoft/responsible-ai-widgets
- https://github.com/microsoft/responsible-ai-widgets/blob/main/docs/explanation-dashboard-README.md
- https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf
- https://towardsdatascience.com/shap-a-reliable-way-to-analyze-your-model-interpretability-874294d30af6

Flood Predictor explainability is important because our main clients will have engineering backgrounds. So, they appreciate being able to understand model predictions. To that end, we use the method of SHapley Additive exPlanations (SHAP), see NEURIPS paper above, and the `azureml-interpret` module. Each feature will be assigned its SHAP value which determines its importance. The importance can be global, over the whole test set, or local, for a few cases of interest where the predictions might not be as expected.

The `azureml-interpret` module is designed to work with sklearn workflows. Therefore, I wrote a wrapper function around all of the pysparl MLlib pipeline stages to adapt `azureml-interpret` to work with MLlib.

## Model registration

### Optimizing metric

The F1 score is chosen as the optimizing metric for skewed data labels.

### Satisficing metrics

- Inference time: The time it takes to make an inference on the standard dataset should be less than 3 seconds. TO BE IMPLEMENTED
    - 10 standard tiles of 256x256 points are to be saved to consistently measure the inference time. TO BE IMPLEMENTED
    - Average the time it takes over all 10 standard tiles. TO BE IMPLEMENTED
