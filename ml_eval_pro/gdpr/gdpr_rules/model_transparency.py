import os

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor, \
    AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, StackingClassifier, \
    StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, \
    RadiusNeighborsRegressor
from sklearn.svm import SVC, NuSVC, SVR, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from xgboost import XGBRegressor

from ml_eval_pro.configuration_manager.configuration_reader.yaml_reader import YamlReader
from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance
from ml_eval_pro.model_transparency.model_transparency_factory import ModelTransparencyFactory


class ModelTransparency(GdprCompliance):
    """

    A class for evaluating model transparency and explain-ability based on GDPR compliance rules.

    Parameters: - model: The trained machine learning model to be evaluated for transparency. - X_test: The feature
    matrix of the test dataset. - y_test: The target values of the test dataset. - problem_type: The type of machine
    learning problem, either 'classification' or 'regression' (default is 'classification'). - X_train: The feature
    matrix of the training dataset (optional). - y_train: The target values of the training dataset (optional).

    Attributes:
    - model: The trained machine learning model.
    - X_test: The feature matrix of the test dataset.
    - y_test: The target values of the test dataset.
    - problem_type: The type of machine learning problem.
    - X_train: The feature matrix of the training dataset (can be None if not provided).
    - y_train: The target values of the training dataset (can be None if not provided).

    """

    def __init__(self, model, X_test, y_test, problem_type='classification', X_train=None, y_train=None,
                 features_description: dict = None, num_of_classes: int = 2, n_bins: int = 5):
        super().__init__(model, X_test, y_test, problem_type, X_train, y_train, features_description, num_of_classes,
                         n_bins)
        self.avg_entropy = self.__calculate_avg_entropy()


    @staticmethod
    def __calculate_entropy(values):
        """Static method to calculate entropy for a given array of values."""
        return -np.sum(values * np.log2(values))

    def __calculate_avg_entropy(self):
        """Calculate the average entropy of SHAP values for the model predictions."""
        explainer = shap.Explainer(self.model.predict, shap.utils.sample(self.X_test, int(self.X_test.shape[0] * 0.1)))
        shap_values = explainer(self.X_test).values
        normalized_shap_values = np.abs(shap_values) / np.sum(np.abs(shap_values))
        epsilon = 1e-10  # To avoid Zero Logarithm problem "-inf"
        normalized_shap_values = np.maximum(normalized_shap_values, epsilon)
        entropies = [self.__calculate_entropy(inner_array) for inner_array in normalized_shap_values[0]]
        return np.mean(entropies)

    def check_significance(self):
        """Check if the average entropy is below a predefined significance threshold."""
        significant = YamlReader(os.path.join(os.path.curdir,
                                              "ml_eval_pro",
                                              "config_files",
                                              "system_config.yaml")).get("thresholds")["shap_significance"]
        if self.avg_entropy < significant:
            return True
        return False

    def check_explain_ability(self):
        """Check the explain-ability of the model based on its type."""

        # model_algorithm = self.model.__dict__["model"]
        #
        # explainable_models = [LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, DecisionTreeClassifier,
        #                       DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor]
        #
        # partially_explainable_models = [KNeighborsClassifier, KNeighborsRegressor,
        #                                 RadiusNeighborsClassifier, RadiusNeighborsRegressor, SVC, NuSVC,
        #                                 SVR, NuSVR, ]
        #
        # complex_models = [RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor,
        #                   AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
        #                   XGBRegressor,
        #                   StackingClassifier, StackingRegressor]
        #
        # if any(isinstance(model_algorithm, model_type) for model_type in explainable_models):
        #     return "A"
        # elif any(isinstance(model_algorithm, model_type) for model_type in partially_explainable_models):
        #     return "B"
        # elif any(isinstance(model_algorithm, model_type) for model_type in complex_models):
        #     return "C"
        # else:
        #     return "I"

        model_transparency = ModelTransparencyFactory.create(self.model)
        return model_transparency.get_model_transparency()

