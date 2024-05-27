# ML-EvalPro

ML-EvalPro is a platform designed to advance ML model evaluation by providing a comprehensive suite of tools for automated testing and analysis for supervised machine learning tasks.

## Prequist Requirements
  1. [Apache Spark](https://spark.apache.org/downloads.html) 
  2. [Ollama](https://ollama.com/)
     * After downloading open anaconda prompt and run this command
       ```bash
       conda install conda-forge::ollama
       ```
      
     * From terminal run this command
       ```bash
       ollama pull llama3
       ```

## Installation

You can install ML-EvalPro using pip:

```bash
pip install ml-eval-pro
```

## Example Script on how to use ML_EvalPro package with mlflow.

This script demonstrates using MLFlow Experiment with LightGBM Regression on California Housing Dataset .

```python
import pandas as pd

from ml_eval_pro.evaluator import Evaluator
import mlflow.xgboost
import lightgbm as lgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Set MLFlow tracking URI and experiment name
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Factory-try")

# Load the California housing dataset
california_data = fetch_california_housing()

# Convert to DataFrame
california_df = pd.DataFrame(data=california_data.data, columns=california_data.feature_names)
california_df["target"] = california_data.target

# Split data into features and target
X = california_df.drop("target", axis=1)
y = california_df["target"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Add context for the dataset
dataset_context = f'this dataset contains information about housing ' \
                  'districts in California. This dataset is often utilized ' \
                  'for regression tasks, particularly for predicting the ' \
                  'median house value for districts based on various features.'

# Add Feature Description
features_description = {'Median House Value': 'This is the target variable, '
                                              'representing the median house value '
                                              'for households in the district.',
                        'Median Income': 'The median income of households in the district, '
                                         'which could be a significant predictor of '
                                         'housing prices.',
                        'Housing Median Age': 'The median age of houses in the district, '
                                              'providing insights into the housing stock\'s '
                                              'age distribution.',
                        'Average Rooms': 'The average number of rooms in houses within the '
                                         'district, which can reflect the size of dwellings.',
                        'Average Bedrooms': 'The average number of bedrooms in houses within '
                                            'the district, indicating the typical household size.',
                        'Population': 'The total population of the district, '
                                      'which may influence housing demand and prices.',
                        'Average Occupancy': 'The average occupancy per household, indicating '
                                             'the housing occupancy rate.'
                                             'providing spatial information.',
                        }

# Define LightGBM parameters for regression
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

with mlflow.start_run(run_name='LightGBM_Regression_California_Test'):
    # Log parameters
    mlflow.log_params(params)

    # Train model
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    model = lgb.train(params, train_data, valid_sets=[test_data])

    # Log model
    model_info = mlflow.lightgbm.log_model(model, "lightgbm_regression_california_model")


# When providing testing and training datasets
evaluator = Evaluator(model_uri=model_info.model_uri,
                      model_type="regression", test_dataset=pd.DataFrame(X_test), test_target=pd.Series(y_test),
                      train_dataset=pd.DataFrame(X_train),
                      train_target=pd.Series(y_train), evaluation_metrics=['MAE'],
                      dataset_context=dataset_context,
                      features_description=features_description)

# Uncomment when using test dataset only 
# evaluator = Evaluator(model_uri=model_info.model_uri,
#                           model_type="regression", test_dataset=pd.DataFrame(X_test), 
#                           test_target=pd.Series(y_test), evaluation_metrics=['MAE'],
#                           dataset_context=dataset_context,
#                           features_description=features_description)

print(evaluator.get_evaluations())
```


