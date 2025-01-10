from abc import abstractmethod

import numpy as np
import pandas as pd

from ml_eval_pro.adverserial_test_cases.adversarial_attack import AdversarialAttack
from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.model.adversarial_attack_substitute_model_factory import \
    AdversarialAttackSubstituteModelFactory
from ml_eval_pro.utils.validation import convert_dataframe_to_numpy


class AdversarialAttackSubstitute(AdversarialAttack):
    def __init__(self, model, model_type: str, test_input_features, test_target_features,
                 dataset_columns, train_input_features=None, train_target_features=None,
                 num_classes=None):
        """
        Create an adversarial attack.
        :param model: the model to be tested against adversarial attacks.
        :param model_type: the model type i.e. Regression, Classification, etc.
        :param test_input_features: the input features of the test dataset.
        :param test_target_features: the target feature(s) of the test dataset.
        :param train_input_features: the input features of the train dataset.
        :param train_target_features: the target feature(s) of the train dataset.
        :param num_classes: the number of classes of a classification model.
        :param dataset_columns: the features name of the dataset.
        :return: the created adversarial attack.
        """
        super().__init__()

        self.model_type = model_type
        self.model = model

        self.test_target_features = convert_dataframe_to_numpy(test_target_features)
        self.train_target_features = convert_dataframe_to_numpy(train_target_features)

        self.train_input_features = train_input_features
        self.test_input_features = test_input_features

        # Get the number of classes from the dataset given if the number of classes is not given
        self.num_classes = num_classes
        if self.model_type == 'classification' and num_classes is None:
            self.num_classes = np.unique(self.test_target_features).shape[0]

        # Splitting the test data if there is no training data
        if train_input_features is None:
            split_size = int(0.8 * self.test_target_features.shape[0])
            self.train_input_features = test_input_features.iloc[:split_size]
            self.train_target_features = test_target_features[:split_size]
            self.test_input_features = test_input_features[split_size:]
            self.test_target_features = test_target_features.iloc[split_size:]

        self.train_model_predictions = self.model.predict(self.train_input_features)
        self.test_model_predictions = self.model.predict(self.test_input_features)

        self.train_input_features = convert_dataframe_to_numpy(self.train_input_features)
        self.test_input_features = convert_dataframe_to_numpy(self.test_input_features)

        self.dataset_columns = dataset_columns

    def get_adversarial_testcases(self):
        """
        Get the adversarial test cases of a model.
        """
        substitute_model = AdversarialAttackSubstituteModelFactory.create(self.model_type,
                                                                          self.train_input_features,
                                                                          self.train_model_predictions)
        attack_model = self.generate_attack_object(substitute_model.generate_model())
        generated_testcases = pd.DataFrame(attack_model.generate(self.test_input_features), columns=self.dataset_columns)
        return self.__generate_adversarial_testcases(generated_testcases)

    def __generate_adversarial_testcases(self, generated_testcases):
        """
        Extract the adversarial test cases from the generated test cases of an attack.
        """
        adversarial_predictions = self.model.predict(generated_testcases)
        threshold = 0 if self.model_type == "classification" else np.std(self.test_model_predictions.unique())

        not_robust_predictions = (abs(adversarial_predictions - self.test_model_predictions)
                                  > threshold)

        not_robust_adversarial_examples = generated_testcases[not_robust_predictions]
        not_robust_true_values = self.test_model_predictions[not_robust_predictions]
        not_robust_predictions = adversarial_predictions[not_robust_predictions]

        self.dataset_columns.extend(["Expected Output", "Model Output"])

        # if the model is robust, return an empty dataframe
        if not_robust_true_values.size == 0:
            self.is_robust = True
            return pd.DataFrame(columns=self.dataset_columns)

        self.is_robust = False
        true_value = not_robust_true_values.reshape((-1, 1))
        not_robust_predictions = not_robust_predictions.reshape((-1, 1))
        adv_test_cases_instances = np.concatenate((not_robust_adversarial_examples, true_value, not_robust_predictions),
                                                  axis=1)

        return pd.DataFrame(adv_test_cases_instances, columns=self.dataset_columns)

    @abstractmethod
    def generate_attack_object(self, substitute_model):
        """
        Generate the adversarial attack object given the substitute model.
        :param substitute_model: the generated substitute model given the data of the original model.
        """
        pass
