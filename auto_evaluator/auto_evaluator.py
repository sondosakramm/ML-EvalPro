import pandas as pd

from auto_evaluator.carbon.carbon_emission.carbon import Carbon
from auto_evaluator.carbon.carbon_emission.carbon_calculator import CarbonCalculator
from auto_evaluator.carbon.inference_time.inference_time import InferenceTime
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
        :param features_description: the features description, where key is the feature and the value is the description.
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

        self.num_classes = -1 if self.model_type == "regression" \
            else get_num_classes(self.model_type, self.test_target)

        print(self.model_type)

        self.features_description = features_description

    def __get_evaluation_metrics(self, target, predictions):
        """
        Calculating the evaluation metrics.
        :param target: the target true values.
        :param predictions: the target prediction values.
        :return: A dictionary of the values.
        """
        res = {}
        for metric in self.evaluation_metrics:
            if self.model_type == 'regression':
                res[metric] = EvaluatorsFactory.get_evaluator(metric, target, predictions).measure()
            elif self.model_type == 'classification':
                res[metric] = EvaluatorsFactory.get_evaluator(metric, target,
                                                              predictions, num_of_classes=self.num_classes).measure()
        return res

    def __get_reliability_diagram(self, data):
        """
        Getting the reliability diagram.
        :param data: the target true dataset.
        :return: the reliability diagram values to be displayed in the graph.
        """
        return EvaluatorsFactory.get_evaluator(f"{self.model_type} reliability evaluation", self.test_target,
                                               self.model_pipeline.predict_proba(data), num_of_classes=self.num_classes).measure()

    def __get_environmental_impact(self):
        """
        Getting the environmental impact values.
        :return: a dictionary of the time inference and the carbon emission values.
        """
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

    def get_evaluations(self):
        """
        Getting all the evaluation values in auto evaluator.
        :return: a dictionary of all the evaluation values in auto evaluator.
        """
        return {
            'evaluation_metrics_test': self.__get_evaluation_metrics(self.test_target, self.test_predictions),

            'evaluation_metrics_train': {} if self.train_dataset is None
            else self.__get_evaluation_metrics(self.test_target, self.test_predictions),

            'reliability_diagram': self.__get_reliability_diagram(self.test_dataset),

            'environmental_impact': self.__get_environmental_impact()
        }
