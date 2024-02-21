import os
import unittest
import joblib
import pandas as pd

from ml_eval_pro.gdpr.gdpr_rules.model_transparency import ModelTransparency


class TestModelTransparency(unittest.TestCase):

    def test_model_transparency(self):

        models = ["linear_regression_boston.joblib", "linear_regression_diabetes.joblib",
                  "logistic_regression_iris.joblib", "logistic_regression_wine.joblib"]

        datasets = ["boston_housing_regression.csv", "diabetes_regression.csv",
                    "iris_classification.csv", "wine_classification.csv"]

        entropy = [0.0018, 0.0023, 0.0423, 0.0057]

        for i in range(len(models)):
            model_file_path = f"../models/{models[i]}"
            model = joblib.load(model_file_path)

            df = pd.read_csv(f"../datasets/{datasets[i]}")
            X = df.iloc[:, :-1]
            Y = df.iloc[:, -1]
            mt = ModelTransparency(X_test=X, y_test=Y, model=model)
            self.assertAlmostEqual(entropy[i], mt.avg_entropy, delta=0.005, msg="The AVG Entropies are not equal")


if __name__ == '__main__':
    unittest.main()
