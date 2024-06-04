from ml_eval_pro.evaluator.adversarial_evaluator import AdversarialEvaluator
from ml_eval_pro.evaluator.env_impact_evaluator import EnvImpactEvaluator
from ml_eval_pro.evaluator.fairness_evaluator import FairnessEvaluator
from ml_eval_pro.evaluator.feature_importance_evaluator import FeatureImportanceEvaluator
from ml_eval_pro.evaluator.gdpr_evaluator import GDPREvaluator
from ml_eval_pro.evaluator.init_evaluator import InitEvaluator
from ml_eval_pro.evaluator.metrics_evaluator import MetricsEvaluator
from ml_eval_pro.evaluator.reliability_evaluator import ReliabilityEvaluator
from ml_eval_pro.evaluator.variance_evaluator import VarianceEvaluator
from ml_eval_pro.model.evaluated_model_factory import EvaluatedModelFactory
from ml_eval_pro.utils.get_config import get_config
from ml_eval_pro.utils.validate_model_type import get_num_classes


class Evaluator:
    def __init__(self, model_uri, problem_type: str,
                 features_description, dataset_context,
                 metrics, test_dataset, target_feature_name, train_dataset=None,
                 spark_session_name=None, spark_feature_col_name="features"):
        """
        Create an instance of the evaluator to get all the evaluation metrics.
        :param model_uri: the model uri.
        :param problem_type: the model problem type.
        :param features_description: the features' description.
        :param dataset_context: a description of the dataset context.
        :param metrics: the evaluation metrics to be calculated.
        :param test_dataset: the test dataset.
        :param target_feature_name: the test feature name.
        :param train_dataset: the train dataset.
        :param spark_session_name: the spark session name(used in spark models only).
        :param spark_feature_col_name: the spark features column name (used in spark models only).
        :return: An instance of the evaluator.
        """
        self.model_pipeline = EvaluatedModelFactory.create(model_uri=model_uri, problem_type=problem_type) \
            if spark_session_name is None else EvaluatedModelFactory.create(model_uri=model_uri,
                                                                            problem_type=problem_type,
                                                                            spark_feature_col_name=spark_feature_col_name,
                                                                            spark_session=spark_session_name)
        self.problem_type = problem_type

        self.features_description = features_description
        self.dataset_context = dataset_context
        self.metrics = metrics

        self.test_target = test_dataset[target_feature_name]
        self.test_dataset = test_dataset.drop([target_feature_name], axis=1)

        self.train_target = train_dataset[target_feature_name] if train_dataset is not None else None
        self.train_dataset = train_dataset.drop([target_feature_name], axis=1) if train_dataset is not None else None

        self.test_predictions = self.model_pipeline.predict(self.test_dataset)
        self.train_predictions = None if self.train_target is None \
            else self.model_pipeline.predict(self.train_dataset)

        self.test_predictions_proba = self.__get_prediction_proba(self.test_dataset)
        self.train_predictions_proba = self.__get_prediction_proba(self.train_dataset)

        self.num_classes = self.__get_num_of_classes()

        self.init_eval = InitEvaluator(self.model_pipeline, self.problem_type, self.num_classes,
                                       self.test_dataset, self.test_target,
                                       self.test_predictions, self.test_predictions_proba,
                                       self.train_dataset, self.train_target,
                                       self.train_predictions, self.train_predictions_proba)

        self.metrics_eval = None
        self.reliability_eval = None
        self.env_impact_eval = None
        self.fairness_eval = None
        self.variance_eval = None
        self.ethical_eval = None
        self.adversarial_eval = None
        self.gdpr_eval = None

    def evaluate(self, **kwargs):
        """
        Evaluate the model from the different evaluation analysis provided.
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
               - bins (int): the number of bins provided for drawing the reliability diagram.
        """
        self.metrics_eval = MetricsEvaluator(self.metrics, self.init_eval)
        self.reliability_eval = ReliabilityEvaluator(self.metrics_eval)
        self.env_impact_eval = EnvImpactEvaluator(self.reliability_eval)
        self.fairness_eval = FairnessEvaluator(self.env_impact_eval)
        self.variance_eval = VarianceEvaluator(self.fairness_eval)
        self.ethical_eval = FeatureImportanceEvaluator(self.features_description, self.dataset_context,
                                                       self.variance_eval)
        self.adversarial_eval = AdversarialEvaluator(self.ethical_eval)
        self.adversarial_eval.evaluate(**get_config(**kwargs))
        self.gdpr_eval = GDPREvaluator(self.features_description, self.dataset_context,
                                       self.adversarial_eval.robust, self.ethical_eval.unethical_features,
                                       self.init_eval)
        self.gdpr_eval.evaluate(**get_config(**kwargs))

    def __dict__(self):
        return {'evaluation_metrics': {
                    'test': self.metrics_eval.test_metrics_values,
                    'train': self.metrics_eval.train_metrics_values,
                },
                'reliability_diagram': self.reliability_eval.reliability_diagram,
                'environmental_impact': {
                    'inference_time_val': self.env_impact_eval.inference_time,
                    'inference_time_summary': self.env_impact_eval.inference_time_summary,
                    'carbon_per_prediction': self.env_impact_eval.carbon_emission_per_prediction,
                    'carbon_prediction_per_kg': self.env_impact_eval.predictions_count_per_kg_carbon,
                    'carbon_summary': self.env_impact_eval.carbon_emission_summary
                },
                'model_bias': {
                    "biased_features": self.fairness_eval.biased_features,
                    "bias_summary": self.fairness_eval.bias_summary,
                    "equalized_odds": self.fairness_eval.equalized_odds,
                    "equalized_odds_summary": self.fairness_eval.equalized_odds_summary
                },
                'model_variance': {
                    'variance_train_value': self.variance_eval.train_variance_value,
                    'high_variance_features': self.variance_eval.high_variance_features,
                    'variance_summary': self.variance_eval.variance_summary
                },
                'ethical_analysis': {
                    'feature_importance': self.ethical_eval.features_importance_scores,
                    'unethical_features': self.ethical_eval.unethical_features
                },

                'adversarial_test_cases': self.adversarial_eval.adversarial_testcases,

                'gdpr_compliance': {
                    'transparency': self.gdpr_eval.model_transparency,
                    'ethical': self.gdpr_eval.model_ethical,
                    'robustness': self.gdpr_eval.model_robustness,
                    'reliability': self.gdpr_eval.model_reliability,
                    },
                }

    def __get_prediction_proba(self, data):
        """
        Getting the prediction probability given the data.
        :param data: the data sample used to get the prediction probability.
        :return: the prediction probability.
        """
        return None if self.problem_type == "regression" or data is None \
            else self.model_pipeline.predict(data, predict_class=False)

    def __get_num_of_classes(self):
        """
        Getting the number of classes given the data and the problem type.
        :return: the number of classes in classification problems and -1 for regression problems.
        """
        return -1 if self.problem_type == "regression" \
            else get_num_classes(self.problem_type, self.test_target)
