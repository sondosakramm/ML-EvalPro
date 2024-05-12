import os

import numpy as np
from art.attacks.evasion.zoo import ZooAttack
from art.estimators.classification.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

from ml_eval_pro.adverserial_test_cases.adversarial_attack import AdversarialAttack
from ml_eval_pro.configuration_manager.configuration_reader.yaml_reader import YamlReader
from ml_eval_pro.utils.optimal_clusters import calculate_optimal_bins


class AdversarialAttackSubstitute(AdversarialAttack):
    """
    A class for generating an adversarial attack by a substitution model.
    """

    def __init__(self, model, model_type: str, test_input_features, test_target_features,
                 train_input_features=None, train_target_features=None, num_classes=None):

        super().__init__(model, model_type, test_input_features,
                         test_target_features, train_input_features,
                         train_target_features, num_classes)

    def generate(self) -> np.ndarray:
        """
        Generate adversarial attack test cases from the train/test data.
        :return: the created adversarial attack test cases.
        """
        global art_model
        global art_attack
        yaml_reader = YamlReader(os.path.join(os.path.curdir, "ml_eval_pro",
                                              "config_files", "system_config.yaml")).get('adversarial_attack')
        if self.model_type == 'classification':
            substitute_model = self.__generate_substitute_model(self.train_model_predictions)

            art_model = SklearnClassifier(substitute_model)

            art_attack = ZooAttack(art_model, confidence=0.0, targeted=False,
                                   learning_rate=float(yaml_reader['learning_rate']),
                                   max_iter=yaml_reader['max_iter'],
                                   binary_search_steps=1, initial_const=1e-3, abort_early=True, use_resize=False,
                                   use_importance=False, nb_parallel=self.num_classes - 1, batch_size=1, variable_h=1e-3)

        elif self.model_type == 'regression':
            # num_bins = yaml_reader['min_num_bins'] if self.train_model_predictions.shape[0] > yaml_reader[
            #     'min_number_instances'] else int(
            #     yaml_reader['num_bins_percentage'] * self.train_model_predictions.shape[0])
            num_bins = calculate_optimal_bins(self.train_model_predictions)
            binning_ranges = np.linspace(min(self.train_model_predictions), max(self.train_model_predictions) + 1,
                                         num_bins + 1)
            training_labels_true = np.digitize(self.train_model_predictions, bins=binning_ranges)

            substitute_model = self.__generate_substitute_model(training_labels_true)

            art_model = SklearnClassifier(substitute_model)

            art_attack = ZooAttack(art_model,
                                   confidence=yaml_reader['confidence'],
                                   targeted=yaml_reader['targeted'],
                                   learning_rate=float(yaml_reader['learning_rate']),
                                   max_iter=yaml_reader['max_iter'],
                                   binary_search_steps=yaml_reader['binary_search_steps'],
                                   initial_const=yaml_reader['initial_const'],
                                   abort_early=yaml_reader['abort_early'],
                                   use_resize=yaml_reader['use_resize'],
                                   use_importance=yaml_reader['use_importance'],
                                   nb_parallel=yaml_reader['nb_parallel'],
                                   batch_size=1,
                                   variable_h=yaml_reader['variable_h'])

        return art_attack.generate(self.test_input_features)

    def get_adversarial_test_cases(self, adversarial_examples, adversarial_predictions):
        """
        Evaluate the model predictions and the adversarial test cases.
        :param adversarial_examples: the generated adversarial examples.
        :param adversarial_predictions: the predictions of the generated adversarial examples.
        :return: the adversarial test cases after evaluation.
        """
        threshold = 0 if self.model_type == "classification" else np.std(self.test_model_predictions.unique())

        not_robust_predictions = (abs(adversarial_predictions - self.test_model_predictions)
                                  > threshold)

        not_robust_adversarial_examples = adversarial_examples[not_robust_predictions]
        not_robust_true_values = self.test_model_predictions[not_robust_predictions]
        not_robust_predictions = adversarial_predictions[not_robust_predictions]

        return not_robust_adversarial_examples, not_robust_true_values, not_robust_predictions

    def evaluate_robustness(self, adversarial_predictions):
        """
        Evaluate the model robustness to adversarial attacks.
        :param adversarial_predictions: the predictions of the generated adversarial examples.
        :return: a flag indicating whether the model is robust to adversarial attacks.
        """
        print("Evaluating the adversarial test cases generated ...")
        threshold = 0 if self.model_type == "classification" else np.std(self.test_model_predictions.unique())
        self.not_robust = np.any(
            abs(adversarial_predictions - self.test_model_predictions) > threshold)

        return not self.not_robust

    def __generate_substitute_model(self, predictions):
        """
        Generating the substitute model.
        :param predictions: the predictions of the local model of the test data.
        :return: the generated substitute model.
        """
        print(f"Generating the substitute model of the {self.model_type} model ...")

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
