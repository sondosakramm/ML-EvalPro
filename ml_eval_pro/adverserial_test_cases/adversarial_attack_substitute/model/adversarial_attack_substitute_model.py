from abc import ABC, abstractmethod

from art.estimators.classification import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline


class AdversarialAttackSubstituteModel(ABC):
    """
    An abstract class for generating a substitute model according to the problem type.
    """

    def __init__(self, train_input_features, train_model_predictions):
        """
        Generate the substitute model object
        :param train_input_features: the input features used for training the model.
        :param train_model_predictions: the model prediction of the given input features.
        """
        self.train_input_features = train_input_features
        self.train_model_predictions = train_model_predictions

    def generate_model(self):
        """
        Generate the substitute model.
        """
        predictions = self.generate_predictions()
        substitute_model = self.__generate_substitute_model(predictions)
        return SklearnClassifier(substitute_model)

    @abstractmethod
    def generate_predictions(self):
        """
        Generate the predictions used for training the substitute model.
        """
        pass

    def __generate_substitute_model(self, predictions):
        """
        Generating the substitute model.
        :param predictions: the predictions of the local model of the test data.
        :return: the generated substitute model.
        """
        rf_pipelines = make_pipeline(RandomForestClassifier())

        rf_hyperparameters = {
            'randomforestclassifier__n_estimators': [50, 100, 300, 500],
            'randomforestclassifier__max_features': ['sqrt', 'log2'],
            'randomforestclassifier__min_samples_leaf': [1, 3, 5, 7],
            'randomforestclassifier__max_depth': [3, 5, 7, 10],
            'randomforestclassifier__criterion': ["gini", "entropy", "log_loss"],
            "randomforestclassifier__class_weight": ["balanced", "balanced_subsample"],
        }

        model = RandomizedSearchCV(
            rf_pipelines,
            rf_hyperparameters,
            cv=3,
            scoring='accuracy'
        )

        model.fit(self.train_input_features, predictions)

        return model.best_estimator_
