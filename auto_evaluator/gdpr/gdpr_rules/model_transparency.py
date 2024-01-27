import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor, \
    AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, StackingClassifier, \
    StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, \
    RadiusNeighborsRegressor
from sklearn.svm import SVC, NuSVC, SVR, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor
from xgboost import XGBRegressor

from auto_evaluator.configuration_manager.configuration_reader.yaml_reader import YamlReader
from auto_evaluator.gdpr.gdpr_compliance import GdprCompliance


class ModelTransparency(GdprCompliance):
    """

    A class for evaluating model transparency and explain-ability based on GDPR compliance rules.

    Parameters:
    - model: The trained machine learning model to be evaluated for transparency.
    - X_test: The feature matrix of the test dataset.
    - y_test: The target values of the test dataset.
    - problem_type: The type of machine learning problem, either 'classification' or 'regression' (default is 'classification').
    - X_train: The feature matrix of the training dataset (optional).
    - y_train: The target values of the training dataset (optional).

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
        self.summary_str = f''

    @staticmethod
    def __calculate_entropy(values):
        """Static method to calculate entropy for a given array of values."""
        return -np.sum(values * np.log2(values))

    def __calculate_avg_entropy(self):
        """Calculate the average entropy of SHAP values for the model predictions."""
        explainer = shap.Explainer(self.model)
        shap_values = explainer.shap_values(self.X_test)
        normalized_shap_values = shap_values / np.sum(np.abs(shap_values))
        epsilon = 1e-10  # To avoid Zero Logarithm problem "-inf"
        normalized_shap_values = np.maximum(normalized_shap_values, epsilon)
        entropies = [self.__calculate_entropy(inner_array) for inner_array in normalized_shap_values[0]]
        return np.mean(entropies)

    def __check_significance(self):
        """Check if the average entropy is below a predefined significance threshold."""
        self.avg_entropy = self.__calculate_avg_entropy()
        significant = YamlReader("../config_files/system_config.yaml").get("thresholds")["shap_significance"]
        if self.avg_entropy < significant:
            return True
        return False

    def __check_explain_ability(self):
        """Check the explain-ability of the model based on its type."""
        explainable_models = [LogisticRegression, Ridge, Lasso, ElasticNet, DecisionTreeClassifier,
                              DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor]

        partially_explainable_models = [KNeighborsClassifier, KNeighborsRegressor,
                                        RadiusNeighborsClassifier, RadiusNeighborsRegressor, SVC, NuSVC,
                                        SVR, NuSVR, ]

        complex_models = [RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor,
                          AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
                          XGBRegressor, StackingClassifier, StackingRegressor]
        self.summary_str += f'{5 * "*"}\tModel Explain-ability/Interpretability\t{5 * "*"}\n'
        try:
            if any(isinstance(self.model, model_type) for model_type in explainable_models):
                self.summary_str += ('The model is explainable because it falls into a category of models known for '
                                     'their interpretability.')
                return True
            elif any(isinstance(self.model, model_type) for model_type in partially_explainable_models):
                self.summary_str += ('The model is partially explainable. '
                                     'While some aspects are interpretable, it may have complex components.')
                return True
            elif any(isinstance(self.model, model_type) for model_type in complex_models):
                self.summary_str += f'The model is complex and less explainable.'
                return True
            else:
                return False
        except Exception as e:
            return f'Model type is not supported {e}'

    def __str__(self):
        """Override of the string representation to provide a summary of model transparency."""
        if self.__check_explain_ability():
            if self.__check_significance():
                self.summary_str += (f'The obtained entropy values are consistently low, averaging around '
                                     f'{self.avg_entropy:.8f}.\n'
                                     f'Low entropy signifies that the model predictions are driven by clear and '
                                     f'distinguishable patterns within the input features.\n')

            else:
                self.summary_str += ('The models interpretability may be more challenging. '
                                     'Which indicates a greater degree of complexity and potential interactions among '
                                     'features in influencing predictions.\n')
        else:
            self.summary_str += f'The model is complex and hard to be explainable and interpret.\n'
        return self.summary_str
