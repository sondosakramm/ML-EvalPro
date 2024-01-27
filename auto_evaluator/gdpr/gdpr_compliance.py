from abc import ABC


class GdprCompliance(ABC):
    """A base class for assessing GDPR compliance of machine learning models."""

    def __init__(self, model, X_test, y_test, problem_type='classification', X_train=None, y_train=None,
                 features_description: dict = None, num_of_classes: int = 2, n_bins: int = 5):
        """
        Parameters:
        - model: The trained machine learning model to be evaluated for GDPR compliance.
        - X_test: The feature matrix of the test dataset.
        - y_test: The target values of the test dataset.
        - problem_type: The type of machine learning problem, either 'classification' or 'regression'
                        (default is 'classification').
        - X_train: The feature matrix of the training dataset (optional).
        - y_train: The target values of the training dataset (optional).
        - features_description: a short description for each feature. (optional)
        - num_of_classes: Number of classes (only for classification tasks) (default=2).
        - n_bins: the number of bins needed (default=5).

        Attributes:
        - model: The trained machine learning model.
        - X_test: The feature matrix of the test dataset.
        - y_test: The target values of the test dataset.
        - problem_type: The type of machine learning problem.
        - X_train: The feature matrix of the training dataset (can be None if not provided).
        - y_train: The target values of the training dataset (can be None if not provided).
        - features_description: a short description for each feature. (can be None if not provided)
        - num_of_classes: Number of classes (only for classification tasks) (default=2).
        - n_bins: the number of bins needed (default=5).
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.problem_type = problem_type
        self.X_train = X_train
        self.y_train = y_train
        self.num_of_classes = num_of_classes
        self.n_bins = n_bins
        self.features_description = features_description