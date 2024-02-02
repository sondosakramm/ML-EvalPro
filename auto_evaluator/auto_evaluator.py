import numpy as np
import pandas as pd

from auto_evaluator.adverserial_test_cases.adversarial_attack_substitute import AdversarialAttackSubstitute
from auto_evaluator.bias.model_bias import ModelBias
from auto_evaluator.carbon.carbon_emission.carbon import Carbon
from auto_evaluator.carbon.carbon_emission.carbon_calculator import CarbonCalculator
from auto_evaluator.carbon.inference_time.inference_time import InferenceTime
from auto_evaluator.ethical_analysis.ethical_analysis import EthicalAnalysis
from auto_evaluator.utils.validate_model_type import check_model_type, get_num_classes
from auto_evaluator.evaluation_metrics.evaluators_factory import EvaluatorsFactory


class AutoEvaluator:

    def __init__(self, model_pipeline, test_dataset: pd.DataFrame, test_target: pd.Series,
                 evaluation_metrics: list, train_dataset: pd.DataFrame = None,
                 train_target: pd.Series = None, features_description: dict = None):
        """
        Create an instance of the auto evaluator to get all the evaluation metrics.
        :param model_pipeline: the model pipeline.
        :param test_dataset: the test dataset features.
        :param test_target: the test dataset target.
        :param evaluation_metrics: the evaluation metrics to be calculated.
        :param train_dataset: the train dataset features.
        :param train_target: the train dataset target.
        :param features_description: the features description,
         where key is the feature and the value is the description.
        :return: An instance of the auto evaluator.
        """
        self.model_pipeline = model_pipeline
        self.model_type = check_model_type(test_target)
        self.test_dataset = test_dataset
        self.test_target = test_target
        self.evaluation_metrics = evaluation_metrics
        self.train_dataset = train_dataset
        self.train_target = train_target

        self.test_predictions = self.model_pipeline.predict(self.test_dataset)
        self.train_predictions = None if self.train_target is None \
            else self.model_pipeline.predict(self.train_dataset)

        self.test_predictions_proba = None if self.model_type == "regression"\
            else self.model_pipeline.predict_proba(self.test_dataset)

        self.train_predictions_proba = None if self.model_type == "regression" or self.train_dataset is None \
            else self.model_pipeline.predict_proba(self.train_dataset)

        self.num_classes = -1 if self.model_type == "regression" \
            else get_num_classes(self.model_type, self.test_target)

        self.features_description = features_description

    def __get_evaluation_metrics(self, target, predictions, predictions_proba):
        """
        Calculating the evaluation metrics.
        :param target: the target true values.
        :param predictions: the target prediction values.
        :param predictions: the target prediction probability values of each class (for classification only).
        :return: A dictionary of the values.
        """
        print("Evaluating the model using the input evaluation metrics ...")
        res = {}
        for metric in self.evaluation_metrics:
            # if self.model_type == 'regression':
            #     res[metric] = EvaluatorsFactory.get_evaluator(metric, target, predictions).measure()
            # elif self.model_type == 'classification':
            if metric == 'expected calibration error' or metric == 'auc':
                res[metric] = EvaluatorsFactory.get_evaluator(metric, target,
                                                              predictions_proba,
                                                              num_of_classes=self.num_classes).measure()
            else:
                res[metric] = EvaluatorsFactory.get_evaluator(metric, target,
                                                              predictions,
                                                              num_of_classes=self.num_classes).measure()
        return res

    def __get_reliability_diagram(self):
        """
        Getting the reliability diagram.
        :return: the reliability diagram values to be displayed in the graph.
        """
        print("Extracting the model reliability diagram ...")
        return EvaluatorsFactory.get_evaluator(f"{self.model_type} reliability evaluation",
                                               self.test_target,
                                               self.test_predictions if self.model_type == 'regression' else
                                               self.test_predictions_proba,
                                               num_of_classes=self.num_classes).measure()

    def __get_environmental_impact(self) -> dict:
        """
        Getting the environmental impact values.
        :return: a dictionary of the time inference and the carbon emission values.
        """
        print("Evaluating the model environmental impact ...")
        inference_time = InferenceTime(self.model_pipeline, self.test_dataset)
        inference_time_val = (inference_time.calc_inference_time_hours()
                              + inference_time.calc_inference_time_minutes()
                              + inference_time.calc_inference_time_seconds())
        inference_time_summary = inference_time.__str__()

        carbon_emission = Carbon(self.model_pipeline, self.test_dataset)
        carbon_emission_calc = CarbonCalculator(carbon_emission)

        carbon_emission_carbon_per_prediction = carbon_emission_calc.calculate_carbon()
        carbon_emission_predictions_per_kg_co = carbon_emission_calc.calculate_predictions()

        carbon_emission_summary = carbon_emission.__str__()

        return {
            'inference_time_val': inference_time_val,
            'inference_time_summary': inference_time_summary,
            'carbon_per_prediction': carbon_emission_carbon_per_prediction,
            'carbon_prediction_per_kg': carbon_emission_predictions_per_kg_co,
            'carbon_summary': carbon_emission_summary
        }

    def __get_model_bias(self) -> dict:
        """
        Calculating the features bias in the model
        :return: a dictionary of the biased features and summary in the model.
        """
        print("Evaluating the model bias ...")
        model_bias = ModelBias(self.model_pipeline, self.model_type, self.test_dataset, self.test_target)
        biased_features = model_bias()
        biased_features_summary = model_bias.__str__()

        return {
            "biased_features": np.array(biased_features)[:, 0],
            "bias_summary": biased_features_summary
        }

    # TODO
    def __get_model_variance(self):
        """
        Calculating the model variance.
        :return:
        """
        print("Evaluating the model variance ...")
        return None

    def __get_feature_importance(self) -> dict:
        """
        Calculating the feature importance values in evaluating the model.
        :return: a dictionary of the biased features and summary in the model.
        """
        print("Evaluating the model ethical issues according to the features importance ...")
        ethical_analysis = EthicalAnalysis(self.model_pipeline, self.test_dataset, self.features_description)
        feature_importance, unethical_features = ethical_analysis()

        return {
            'feature_importance': feature_importance,
            'feature_ethnicity': unethical_features
        }

    def __get_adversarial_test_cases(self) -> pd.DataFrame:
        """
        Getting the adversarial test cases of the model.
        :return: a data frame containing the adversarial test cases,
         the expected output, and the model output.
        """
        print("Generating the model adversarial test cases ...")
        adversarial_attack = (
            AdversarialAttackSubstitute(self.model_pipeline, self.model_type,
                                        self.test_dataset, self.test_target,
                                        num_classes=self.num_classes) if self.train_target is None else
            AdversarialAttackSubstitute(self.model_pipeline, self.model_type,
                                        self.test_dataset, self.test_target,
                                        train_input_features=self.train_dataset,
                                        train_target_features=self.train_target,
                                        num_classes=self.num_classes))

        adversarial_examples_generated = adversarial_attack.generate()
        dataset_columns = list(self.test_dataset.columns)

        adversarial_examples_generated_df = pd.DataFrame(adversarial_examples_generated,
                                                         columns=dataset_columns)

        adversarial_examples_predictions = self.model_pipeline.predict(adversarial_examples_generated_df)

        adv_test_cases, true_value, predicted_value = adversarial_attack.get_adversarial_test_cases(
            adversarial_examples_generated,
            adversarial_examples_predictions)

        true_value = true_value.reshape((-1, 1))
        predicted_value = predicted_value.reshape((-1, 1))
        adv_test_cases_instances = np.concatenate((adv_test_cases, true_value, predicted_value), axis=1)
        dataset_columns.extend(["Expected Output", "Model Output"])

        return pd.DataFrame(adv_test_cases_instances, columns=dataset_columns)

    # TODO
    def __get_model_gdpr_compliance(self):
        """
        Calculating the model GDPR Compliance.
        :return:
        """
        print("Evaluating the model GDPR Compliance ...")
        return None

    def get_evaluations(self) -> dict:
        """
        Getting all the evaluation values in auto evaluator.
        :return: a dictionary of all the evaluation values in auto evaluator.
        """
        return {
            'evaluation_metrics_test': self.__get_evaluation_metrics(self.test_target, self.test_predictions, self.test_predictions_proba),

            'evaluation_metrics_train': {} if self.train_dataset is None
            else self.__get_evaluation_metrics(self.train_target, self.train_predictions, self.train_predictions_proba),

            'reliability_diagram': self.__get_reliability_diagram(),

            'environmental_impact': self.__get_environmental_impact(),

            'model_bias': self.__get_model_bias(),

            'model_variance': self.__get_model_variance(),

            'ethical_analysis': self.__get_feature_importance(),

            'adversarial_test_cases': self.__get_adversarial_test_cases(),

            'gdpr_compliance': self.__get_model_gdpr_compliance()

        }
