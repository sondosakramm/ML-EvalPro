from ml_eval_pro.carbon.carbon_emission.carbon import Carbon
from ml_eval_pro.carbon.carbon_emission.carbon_calculator import CarbonCalculator
from ml_eval_pro.carbon.inference_time.inference_time import InferenceTime
from ml_eval_pro.evaluator.base_evaluator import BaseEvaluator
from ml_eval_pro.evaluator.interface_evaluator import InterfaceEvaluator
from ml_eval_pro.summary.modules_summary.carbon_summary import CarbonSummary
from ml_eval_pro.summary.modules_summary.inference_summary import InferenceSummary


class EnvImpactEvaluator(BaseEvaluator):
    """
    A class of the environmental impact evaluator.
    """
    def __init__(self, evaluator: InterfaceEvaluator):
        """
        Initializing the environmental impact evaluator.
        :param evaluator: an instance of the evaluator used to initialize the main parameters and evaluate it.
        """
        super().__init__(evaluator)

        self.__inference_time = None
        self.__carbon_emission_per_prediction = None
        self.__predictions_count_per_kg_carbon = None
        self.__inference_time_summary = None
        self.__carbon_emission_summary = None

    def evaluate(self, **kwargs):
        """
        Evaluate the environmental impact from the different evaluation analysis provided.
        :param kwargs: the keys needed to evaluate the environmental impact.
        """
        super().evaluate(**kwargs)

        print("Evaluating the model environmental impact ...")

        cpu_name = kwargs['cpu_name']
        cpu_speed = kwargs['cpu_speed']
        cpu_value = kwargs['cpu_value']
        energy_name = kwargs['energy_name']
        energy_value = kwargs['energy_value']

        inference_time = InferenceTime(self.model_pipeline, self.test_dataset)
        self.__inference_time = (inference_time.calc_inference_time_hours() * 360
                                 + inference_time.calc_inference_time_minutes() * 60
                                 + inference_time.calc_inference_time_seconds())

        carbon_emission = Carbon(self.model_pipeline, self.test_dataset,
                                 cpu_name=cpu_name, cpu_speed=cpu_speed,
                                 cpu_value=cpu_value, energy_name=energy_name, energy_value=energy_value)

        carbon_emission_calc = CarbonCalculator(carbon_emission)

        self.__carbon_emission_per_prediction = carbon_emission_calc.calculate_carbon()
        self.__predictions_count_per_kg_carbon = carbon_emission_calc.calculate_predictions()

        self.__inference_time_summary = InferenceSummary(inference_time).get_summary()
        self.__carbon_emission_summary = CarbonSummary(carbon_emission).get_summary()

    @property
    def inference_time(self):
        return self.__inference_time

    @property
    def carbon_emission_per_prediction(self):
        return self.__carbon_emission_per_prediction

    @property
    def predictions_count_per_kg_carbon(self):
        return self.__predictions_count_per_kg_carbon

    @property
    def inference_time_summary(self):
        return self.__inference_time_summary

    @property
    def carbon_emission_summary(self):
        return self.__carbon_emission_summary
