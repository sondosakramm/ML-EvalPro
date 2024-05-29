import numpy as np
import shap
from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance
from ml_eval_pro.transparency.transparency_factory import TransparencyFactory


class ModelTransparency(GdprCompliance):
    """

    A class for evaluating model transparency and explain-ability based on GDPR compliance rules.

    Parameters:
    - model: The trained machine learning model to be evaluated for transparency. - test_dataset: The
    feature matrix of the test dataset.
    - test_target: The target values of the test dataset.
    - problem_type: The
    - shap_threshold: The SHAP (SHapley Additive exPlanations) value threshold for model interpretability.
    type of machine learning problem, either 'classification' or 'regression' (default is 'classification'). -
    train_dataset: The feature matrix of the training dataset (optional).
    - train_target: The target values of the
    training dataset (optional).

    Attributes:
    - model: The trained machine learning model.
    - test_dataset: The feature matrix of the test dataset.
    - test_target: The target values of the test dataset.
    - problem_type: The type of machine learning problem.
    - train_dataset: The feature matrix of the training dataset (can be None if not provided).
    - train_target: The target values of the training dataset (can be None if not provided).

    """

    def __init__(self, model, X_test, y_test, shap_threshold: float = 0.05, problem_type='classification', X_train=None,
                 y_train=None, features_description: dict = None, num_of_classes: int = 2, n_bins: int = 5):

        super().__init__(model=model, X_test=X_test, y_test=y_test, problem_type=problem_type,
                         X_train=X_train, y_train=y_train, features_description=features_description,
                         num_of_classes=num_of_classes,
                         n_bins=n_bins)
        self.avg_entropy = 0.1
        self.shap_threshold = shap_threshold


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
        if self.avg_entropy < self.shap_threshold:
            return True
        return False

    def check_explain_ability(self):
        """Check the explain-ability of the model based on its type."""
        model_transparency = TransparencyFactory.create(self.model)
        return model_transparency.get_model_transparency()
