import numpy as np
from art.attacks.evasion.zoo import ZooAttack
from art.estimators.classification.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

from auto_evaluator.adverserial_test_cases.adversarial_attack import AdversarialAttack
from auto_evaluator.configuration_manager.configuration_reader.yaml_reader import YamlReader
from auto_evaluator.evaluation_metrics.evaluators_factory import EvaluatorsFactory


class AdversarialAttackSubstitute(AdversarialAttack):
    """
    A class for generating an adversarial attack by a substitution model.
    """

    def generate(self) -> np.ndarray:
        """
        Generate adversarial attack test cases from the train/test data.
        :return: the created adversarial attack test cases.
        """
        global art_model
        global art_attack
        yaml_reader = YamlReader('../config_files/system_config.yaml').get('adversarial_attack')
        if self.model_type == 'classification':
            print("Classification model ....")

            substitute_model = self.__generate_substitute_model(self.train_model_predictions)

            art_model = SklearnClassifier(substitute_model)

            art_attack = ZooAttack(art_model, confidence=0.0, targeted=False,
                                   learning_rate=float(yaml_reader['learning_rate']),
                                   max_iter=yaml_reader['max_iter'],
                                   binary_search_steps=1, initial_const=1e-3, abort_early=True, use_resize=False,
                                   use_importance=False, nb_parallel=self.num_classes - 1, batch_size=1, variable_h=0.2)

        elif self.model_type == 'regression':
            print("Regression model ....")

            num_bins = yaml_reader['min_num_bins'] if self.train_model_predictions.shape[0] > yaml_reader['min_number_instances'] else int(
                yaml_reader['num_bins_percentage'] * self.train_model_predictions.shape[0])
            binning_ranges = np.linspace(min(self.train_model_predictions), max(self.train_model_predictions) + 1,
                                         num_bins + 1)
            training_labels_true = np.digitize(self.train_model_predictions, bins=binning_ranges)

            substitute_model = self.__generate_substitute_model(training_labels_true)

            art_model = SklearnClassifier(substitute_model)

            art_attack = ZooAttack(art_model, confidence=0.0, targeted=False,
                                   learning_rate=float(yaml_reader['learning_rate']),
                                   max_iter=yaml_reader['max_iter'],
                                   binary_search_steps=1, initial_const=1e-3, abort_early=True, use_resize=False,
                                   use_importance=False, nb_parallel=yaml_reader['nb_parallel'], batch_size=1,
                                   variable_h=0.2)

        return art_attack.generate(self.test_input_features)

    def evaluate_robustness(self, adversarial_examples):
        """
        Evaluate the model robustness to adversarial attacks.
        :param adversarial_examples: the generated adversarial examples.
        :return: a flag indicating whether the model is robust to adversarial attacks.
        """

        global test_predictions_eval
        global attacks_eval
        yaml_reader = YamlReader('../config_files/system_config.yaml').get('adversarial_attack')

        print("Evaluating the adversarial test cases generated ...")

        self.adversarial_predictions = self.model.predict(adversarial_examples)

        if self.model_type == "regression":
            attacks_eval = EvaluatorsFactory.get_evaluator("mape", self.test_target_features,
                                                           self.adversarial_predictions).measure()
            test_predictions_eval = EvaluatorsFactory.get_evaluator("mape", self.test_target_features,
                                                                    self.test_model_predictions).measure()
        elif self.model_type == "classification":
            attacks_eval = EvaluatorsFactory.get_evaluator("f1 score", self.test_target_features,
                                                           self.adversarial_predictions).measure()
            test_predictions_eval = EvaluatorsFactory.get_evaluator("f1 score", self.test_target_features,
                                                                    self.test_model_predictions).measure()

        self.not_robust = np.any(abs(self.adversarial_predictions - self.test_model_predictions) > yaml_reader['threshold'])
        if self.not_robust:
            return False
        else:
            return True

    def __generate_substitute_model(self, predictions):
        """
        Generating the substitute model.
        :param predictions: the predictions of the local model of the test data.
        :return: the generated substitute model.
        """
        print("Generating the substitute model ...")

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
