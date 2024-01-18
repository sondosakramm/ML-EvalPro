from abc import abstractmethod, ABC
from auto_evaluator.configuration_manager.configuration_reader.yaml_reader import YamlReader
from auto_evaluator.evaluation_metrics.classification.class_evaluation.accuracy import Accuracy
from auto_evaluator.evaluation_metrics.regression.mae import MAE


class ModelVariance(ABC):
    """
        An abstract class for measuring the model variance.

        Attributes:
        - yaml_reader : YamlReader
            An instance of the YamlReader class for reading system configuration from a YAML file.

        - model : object
            The machine learning model being evaluated.

        - X_test : array-like or pd.DataFrame
            The feature matrix of the test dataset.

        - y_test : array-like or pd.Series
            The true labels or target values for the test dataset.

        - X_train : array-like or pd.DataFrame or None
            The feature matrix of the training dataset.

        - y_train : array-like or pd.Series or None
            The true labels or target values for the training dataset.

        - problem_type : str
            The type of problem, either 'regression' or 'classification'.

        - metric : str or object
            The evaluation metric used for assessing model performance.
    """

    def __init__(self, model, X_test, y_test, X_train=None, y_train=None, problem_type='regression', metric='MAE'):
        """
            Initialize a ModelVariance object for assessing the performance of a machine learning model.

            Parameters:
            - model : object
                The machine learning model to be evaluated.

            - X_test : array-like or pd.DataFrame
                The feature matrix of the test dataset.

            - y_test : array-like or pd.Series
                The true labels or target values for the test dataset.

            - X_train : array-like or pd.DataFrame or None, optional
                The feature matrix of the training dataset. Defaults to None.

            - y_train : array-like or pd.Series or None, optional
                The true labels or target values for the training dataset. Defaults to None.

            - problem_type : str, optional (default='regression')
                The type of problem, either 'regression' or 'classification'.

            - metric : str or object, optional (default='MAE')
                The evaluation metric to be used.
        """
        self.yaml_reader = YamlReader('../config_files/system_config.yaml')
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.problem_type = problem_type
        self.metric = metric

    @abstractmethod
    def calculate_variance(self):
        """
        Abstract method for calculating the model variance.

        This method must be implemented in the derived class.
        """
        pass

    def calculate_errors(self, y, predictions):
        """
        Calculate evaluation metric errors based on the problem type.

        Parameters:
        - y : array-like or pd.Series
            True labels or target values.

        - predictions : array-like or pd.Series
            Predicted values.

        Returns:
            - float: The evaluation metric error.
        """
        if self.problem_type == 'classification':
            # TODO: Use evaluation metric factory based on metric param.
            return Accuracy(y, predictions).measure()
        elif self.problem_type == 'regression':
            # TODO: Use evaluation metric factory based on metric param.
            return MAE(y, predictions).measure()
