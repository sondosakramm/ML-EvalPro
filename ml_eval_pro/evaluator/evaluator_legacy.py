import pandas as pd

from ml_eval_pro.adverserial_test_cases.adversarial_attack_factory import AdversarialAttackFactory
from ml_eval_pro.carbon.carbon_emission.carbon import Carbon
from ml_eval_pro.carbon.carbon_emission.carbon_calculator import CarbonCalculator
from ml_eval_pro.carbon.inference_time.inference_time import InferenceTime
from ml_eval_pro.ethical_analysis.ethical_analysis import EthicalAnalysis
from ml_eval_pro.evaluation_metrics.evaluators_factory import EvaluatorsFactory
from ml_eval_pro.gdpr.gdpr_rules.model_ethical import ModelEthical
from ml_eval_pro.gdpr.gdpr_rules.model_reliability import ModelReliability
from ml_eval_pro.gdpr.gdpr_rules.model_robustness import ModelRobustness
from ml_eval_pro.gdpr.gdpr_rules.model_transparency import ModelTransparency
from ml_eval_pro.model.evaluated_model_factory import EvaluatedModelFactory
from ml_eval_pro.model_fairness.model_fairness_factory import ModelFairnessFactory
from ml_eval_pro.summary.modules_summary.bias_summary import BiasSummary
from ml_eval_pro.summary.modules_summary.carbon_summary import CarbonSummary
from ml_eval_pro.summary.modules_summary.equalized_odds_summary import EqualizedOddsSummary
from ml_eval_pro.summary.modules_summary.inference_summary import InferenceSummary
from ml_eval_pro.summary.modules_summary.model_ethical_summary import ModelEthicalSummary
from ml_eval_pro.summary.modules_summary.model_reliability_summary import ModelReliabilitySummary
from ml_eval_pro.summary.modules_summary.model_robustness_summary import ModelRobustnessSummary
from ml_eval_pro.summary.modules_summary.model_transparency_summary import ModelTransparencySummary
from ml_eval_pro.summary.modules_summary.variance_summary import VarianceSummary
from ml_eval_pro.utils.validate_model_type import get_num_classes
from ml_eval_pro.variance.model_var.model_variance_factory import ModelVarianceFactory


class EvaluatorLegacy:

    def __init__(self, model_uri, model_type: str, test_dataset: pd.DataFrame, test_target: pd.Series,
                 evaluation_metrics: list, features_description: dict, dataset_context: str,
                 train_dataset: pd.DataFrame = None, train_target: pd.Series = None,
                 spark_session_name=None, spark_feature_col_name="features", **kwargs):
        """
        Create an instance of the auto evaluator to get all the evaluation metrics.
        :param model_uri: the model uri.
        :param model_type: the model problem type.
        :param test_dataset: the test dataset features.
        :param test_target: the test dataset target.
        :param evaluation_metrics: the evaluation metrics to be calculated.
        :param features_description: the features' description.
        :param dataset_context: a description of the dataset context.
        :param train_dataset: the train dataset features.
        :param train_target: the train dataset target.
        :param spark_session_name: the spark session name(used in spark models only).
        :param spark_feature_col_name: the spark features column name (used in spark models only).
        :param kwargs: Additional configuration parameters, including:
                       - shap_threshold (float): The SHAP (SHapley Additive exPlanations) value threshold for
                                                 model interpretability.
                       - ece_threshold (float): The Expected Calibration Error (ECE) threshold for evaluating
                                                model calibration.
                       - bias_threshold (float): The bias threshold for assessing model bias.
                       - variance_threshold (float): The variance threshold for assessing model variance.
                       - variance_step_size (float): The step size to be used when adjusting variance.
                       - cpu_name (str): The name of the CPU being used.
                       - cpu_speed (float): The speed of the CPU in GHz.
                       - cpu_value (float): The value or performance index of the CPU.
                       - energy_name (str): The name or type of energy resource being used.
                       - energy_value (float): The value or consumption rate of the energy resource.
                       - llama_model (str): The name of the llama model to get unethical features.
        :return: An instance of the auto evaluator.
        """
        self.spark_feature_col_name = spark_feature_col_name
        self.model_pipeline = EvaluatedModelFactory.create(model_uri=model_uri, problem_type=model_type) \
            if spark_session_name is None else EvaluatedModelFactory.create(model_uri=model_uri,
                                                                            problem_type=model_type,
                                                                            spark_feature_col_name=spark_feature_col_name,
                                                                            spark_session=spark_session_name)
        self.model_type = model_type
        self.test_dataset = test_dataset
        self.test_target = test_target
        self.evaluation_metrics = evaluation_metrics
        self.train_dataset = train_dataset
        self.train_target = train_target

        self.test_predictions = self.model_pipeline.predict(self.test_dataset)
        self.train_predictions = None if self.train_target is None \
            else self.model_pipeline.predict(self.train_dataset)

        self.test_predictions_prob = None if self.model_type == "regression" \
            else self.model_pipeline.predict(self.test_dataset, predict_class=False)

        self.train_predictions_prob = None if self.model_type == "regression" or self.train_dataset is None \
            else self.model_pipeline.predict(self.train_dataset, predict_class=False)

        self.num_classes = -1 if self.model_type == "regression" \
            else get_num_classes(self.model_type, self.test_target)

        self.features_description = features_description
        self.dataset_context = dataset_context
        self.kwargs = kwargs

        self.shap_threshold = self.kwargs.get('shap_threshold', 0.05)
        self.ece_threshold = self.kwargs.get('ece_threshold', 0.05)
        self.bias_threshold = self.kwargs.get('bias_threshold', 0.15)
        self.variance_threshold = self.kwargs.get('variance_threshold', 0.15)
        self.variance_step_size = self.kwargs.get('variance_step_size', 0.5)
        self.cpu_name = self.kwargs.get('cpu_name', 'Xeon')
        self.cpu_speed = self.kwargs.get('cpu_speed', 2.2)
        self.cpu_value = self.kwargs.get('cpu_value', 0.061)
        self.energy_name = self.kwargs.get('energy_name', 'Fuel')
        self.energy_value = self.kwargs.get('energy_value', 0.865)
        self.llama_model = self.kwargs.get('llama_model', 'llama3')
        self.bins = self.kwargs.get('bins', 10)

        # TODO: to be removed after proper code refactoring
        self.unethical_features = None
        self.robustness = None

    def _get_evaluation_metrics(self, target, predictions, predictions_prob):
        """
        Calculating the evaluation metrics.
        :param target: the target true values.
        :param predictions: the target prediction values.
        :param predictions_prob: the target prediction probability values of each class (for classification only).
        :return: A dictionary of the values.
        """
        print("Evaluating the model using the input evaluation metrics ...")
        res = {}
        for metric in self.evaluation_metrics:
            if metric == 'Expected Calibration Error' or metric == 'AUC':
                if self.model_type == 'regression':
                    res[metric] = EvaluatorsFactory.create(metric, target,
                                                           predictions_prob).measure()
                else:
                    res[metric] = EvaluatorsFactory.create(metric, target,
                                                           predictions_prob,
                                                           num_of_classes=self.num_classes).measure()

            else:
                if self.model_type == 'regression':
                    res[metric] = EvaluatorsFactory.create(metric, target,
                                                           predictions).measure()
                else:
                    res[metric] = EvaluatorsFactory.create(metric, target,
                                                           predictions,
                                                           num_of_classes=self.num_classes).measure()

        return res

    def _get_reliability_diagram(self):
        """
        Getting the reliability diagram.
        :return: the reliability diagram values to be displayed in the graph.
        """
        print("Extracting the model reliability diagram ...")
        if self.model_type == 'regression':
            return EvaluatorsFactory.create(f"{self.model_type} reliability evaluation",
                                            self.test_target,
                                            self.test_predictions,
                                            n_bins=self.bins).measure()
        else:
            return EvaluatorsFactory.create(f"{self.model_type} reliability evaluation",
                                            self.test_target,
                                            self.test_predictions_prob,
                                            num_of_classes=self.num_classes).measure()


    def _get_environmental_impact(self) -> dict:
        """
        Getting the environmental impact values.
        :return: a dictionary of the time inference and the carbon emission values.
        """
        print("Evaluating the model environmental impact ...")
        inference_time = InferenceTime(self.model_pipeline, self.test_dataset)
        inference_time_val = (inference_time.calc_inference_time_hours() * 360
                              + inference_time.calc_inference_time_minutes() * 60
                              + inference_time.calc_inference_time_seconds())

        carbon_emission = Carbon(self.model_pipeline, self.test_dataset, cpu_name=self.cpu_name,
                                 cpu_speed=self.cpu_speed, cpu_value=self.cpu_value,
                                 energy_name=self.energy_name, energy_value=self.energy_value)
        carbon_emission_calc = CarbonCalculator(carbon_emission)

        carbon_emission_carbon_per_prediction = carbon_emission_calc.calculate_carbon()
        carbon_emission_predictions_per_kg_co = carbon_emission_calc.calculate_predictions()

        return {
            'inference_time_val': inference_time_val,
            'inference_time_summary': InferenceSummary(inference_time).get_summary(),
            'carbon_per_prediction': carbon_emission_carbon_per_prediction,
            'carbon_prediction_per_kg': carbon_emission_predictions_per_kg_co,
            'carbon_summary': CarbonSummary(carbon_emission).get_summary()
        }

    def _get_model_bias(self) -> dict:
        """
        Calculating the features bias in the model
        :return: a dictionary of the biased features and summary in the model.
        """
        print("Evaluating the model bias and fairness...")
        eval_metric = 'MAPE' if self.model_type == 'regression' else 'Accuracy'

        model_bias = ModelFairnessFactory.create("bias",
                                                 model=self.model_pipeline,
                                                 model_type=self.model_type,
                                                 data=self.test_dataset,
                                                 target=self.test_target,
                                                 evaluation_metrics=[eval_metric],
                                                 threshold=self.bias_threshold)

        biased_features = model_bias.get_model_fairness()

        if self.model_type == 'regression':
            return {
                "biased_features": list(biased_features.keys()),
                "bias_summary": BiasSummary(biased_features, self.bias_threshold).get_summary(),
                "equalized_odds": 'Not supported in regression...'
            }

        model_equalized_odds = ModelFairnessFactory.create("equalized odds",
                                                           model=self.model_pipeline,
                                                           model_type=self.model_type,
                                                           data=self.test_dataset,
                                                           target=self.test_target)

        equalized_odds_features_vals = model_equalized_odds.get_model_fairness()

        return {
            "biased_features": list(biased_features.keys()),
            "bias_summary": BiasSummary(biased_features, self.bias_threshold).get_summary(),
            "equalized_odds": EqualizedOddsSummary(equalized_odds_features_vals).get_summary()
        }

    def _get_model_variance(self):
        """
        Calculating the model variance.
        :return: the model variance and its summary.
        """
        print("Evaluating the model variance ...")
        eval_metric = 'MAE' if self.model_type == 'regression' else 'Accuracy'
        variance_res = {}
        if self.train_target is not None:
            model_variance = ModelVarianceFactory.create(variance_type='train_test_data',
                                                         model=self.model_pipeline,
                                                         model_type=self.model_type,
                                                         train_dataset=self.train_dataset,
                                                         train_target=self.train_target,
                                                         test_dataset=self.test_dataset,
                                                         target=self.test_target,
                                                         evaluation_metric=eval_metric,
                                                         threshold=self.variance_threshold)
            variance_res['variance_train_value'] = model_variance.calculate_variance()
            variance_res['variance_train_summary'] = VarianceSummary(model_variance).get_summary()
        else:
            model_variance = ModelVarianceFactory.create(variance_type='test_data',
                                                         model=self.model_pipeline,
                                                         model_type=self.model_type,
                                                         data=self.test_dataset,
                                                         target=self.test_target,
                                                         evaluation_metric=eval_metric,
                                                         threshold=self.variance_threshold,
                                                         step_size=self.variance_step_size)
            model_variance.calculate_variance()
            variance_res['high_variant_features'] = model_variance.get_diff()
            variance_res['variance_features_summary'] = VarianceSummary(model_variance).get_summary()

        return variance_res

    def _get_feature_importance(self) -> dict:
        """
        Calculating the feature importance values in evaluating the model.
        :return: a dictionary of the biased features and summary in the model.
        """
        print("Evaluating the model ethical issues according to the features importance ...")
        ethical_analysis = EthicalAnalysis(self.model_pipeline, self.test_dataset,
                                           self.features_description, self.dataset_context)
        feature_importance, unethical_features = ethical_analysis(llama_model=self.llama_model)

        # TODO: to be removed after proper code refactoring
        self.unethical_features = unethical_features

        return {
            'feature_importance': feature_importance,
            'feature_ethnicity': unethical_features
        }

    def _get_adversarial_test_cases(self) -> pd.DataFrame:
        """
        Getting the adversarial test cases of the model.
        :return: a data frame containing the adversarial test cases,
         the expected output, and the model output.
        """
        print("Generating the model adversarial test cases ...")
        dataset_columns = list(self.test_dataset.columns)

        adversarial_attack_model = (
            AdversarialAttackFactory.create("substitute_zoo", self.model_pipeline, self.model_type,
                                            self.test_dataset, self.test_target, dataset_columns,
                                            num_classes=self.num_classes) if self.train_target is None else

            AdversarialAttackFactory.create("substitute_zoo", self.model_pipeline, self.model_type,
                                            self.test_dataset, self.test_target, dataset_columns,
                                            train_input_features=self.train_dataset,
                                            train_target_features=self.train_target,
                                            num_classes=self.num_classes))

        adversarial_testcases = adversarial_attack_model.get_adversarial_testcases()
        self.robustness = adversarial_attack_model.is_robust

        return adversarial_testcases

    def _get_model_gdpr_compliance(self):
        """
        Evaluating the model GDPR Compliance.
        :return: the model GDPR Compliance summary.
        """
        print("Evaluating the model GDPR Compliance ...")

        # TODO: Needs code refactoring
        model_ethical = ModelEthical(self.model_pipeline, features_description=self.features_description,
                                     dataset_context=self.dataset_context,
                                     X_test=self.test_dataset,
                                     unethical_features=self.unethical_features,
                                     llama_model=self.llama_model)

        model_reliability = ModelReliability(model=self.model_pipeline,
                                             X_test=self.test_dataset,
                                             y_test=self.test_target,
                                             problem_type=self.model_type,
                                             num_of_classes=self.num_classes)

        # TODO: Needs code refactoring
        model_robustness = ModelRobustness(model=self.model_pipeline,
                                           X_test=self.test_dataset,
                                           y_test=self.test_target,
                                           X_train=self.train_dataset,
                                           y_train=self.train_target,
                                           problem_type=self.model_type,
                                           robustness=self.robustness)

        model_transparency = ModelTransparency(model=self.model_pipeline,
                                               X_test=self.test_dataset,
                                               y_test=self.test_target,
                                               problem_type=self.model_type,
                                               shap_threshold=self.shap_threshold)

        return (ModelReliabilitySummary(model_reliability, self.ece_threshold).get_summary() +
                '\n' + ModelEthicalSummary(model_ethical).get_summary() +
                '\n' + ModelRobustnessSummary(model_robustness).get_summary() +
                '\n' + ModelTransparencySummary(model_transparency).get_summary())

    def _get_machine_unlearning_ability(self):
        """
        Evaluating the model machine unlearning ability.
        :return: the model machine unlearning ability summary.
        """
        return 'Testing machine unlearning ability is not Supported yet!'

    def get_evaluations(self) -> dict:
        """
        Getting all the evaluation values in auto evaluator.
        :return: a dictionary of all the evaluation values in auto evaluator.
        """
        return {
            'evaluation_metrics_test': self._get_evaluation_metrics(self.test_target, self.test_predictions,
                                                                    self.test_predictions_prob),

            'evaluation_metrics_train': {} if self.train_dataset is None
            else self._get_evaluation_metrics(self.train_target, self.train_predictions, self.train_predictions_prob),

            'reliability_diagram': self._get_reliability_diagram(),

            'environmental_impact': self._get_environmental_impact(),

            'model_bias': self._get_model_bias(),

            'model_variance': self._get_model_variance(),

            'ethical_analysis': self._get_feature_importance(),

            'adversarial_test_cases': self._get_adversarial_test_cases(),

            'gdpr_compliance': self._get_model_gdpr_compliance(),
            #
            # 'machine_unlearning': self.__get_machine_unlearning_ability()

        }
