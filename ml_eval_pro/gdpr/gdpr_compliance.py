from abc import ABC


class GdprCompliance(ABC):
    """A base class for assessing GDPR compliance of machine learning models."""

    def __init__(self, model=None, X_test=None, y_test=None, problem_type='classification', X_train=None, y_train=None,
                 features_description: dict = None, dataset_context: str = None, num_of_classes: int = 2, n_bins: int = 5,
                 unethical_features=None, adversarial_attack_model=None, adversarial_testcases=None, llama_model=None):
        """
        Parameters:
        - model: The trained machine learning model to be evaluated for GDPR compliance (optional).
        - test_dataset: The feature matrix of the test dataset (optional).
        - test_target: The target values of the test dataset (optional).
        - problem_type: The type of machine learning problem, either 'classification' or 'regression'
                        (default is 'classification').
        - train_dataset: The feature matrix of the training dataset (optional).
        - train_target: The target values of the training dataset (optional).
        - features_description: a short description for each feature. (optional)
        - dataset_context: a description for the dataset context. (optional)
        - num_of_classes: Number of classes (only for classification tasks) (default=2).
        - n_bins: the number of bins needed (default=5).

        Attributes:
        - model: The trained machine learning model (can be None if not provided)..
        - test_dataset: The feature matrix of the test dataset (can be None if not provided)..
        - test_target: The target values of the test dataset (can be None if not provided)..
        - problem_type: The type of machine learning problem.
        - train_dataset: The feature matrix of the training dataset (can be None if not provided).
        - train_target: The target values of the training dataset (can be None if not provided).
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
        self.dataset_context = dataset_context
        self.llama_model = llama_model

        # TODO: to be removed after proper code refactoring
        self.unethical_features = unethical_features
        self.adversarial_attack_model = adversarial_attack_model
        self.adversarial_testcases = adversarial_testcases
