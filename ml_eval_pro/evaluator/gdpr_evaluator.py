from ml_eval_pro.evaluator.base_evaluator import BaseEvaluator
from ml_eval_pro.evaluator.interface_evaluator import InterfaceEvaluator
from ml_eval_pro.gdpr.gdpr_rules.model_ethical import ModelEthical
from ml_eval_pro.gdpr.gdpr_rules.model_reliability import ModelReliability
from ml_eval_pro.gdpr.gdpr_rules.model_robustness import ModelRobustness
from ml_eval_pro.gdpr.gdpr_rules.model_transparency import ModelTransparency
from ml_eval_pro.summary.modules_summary.model_ethical_summary import ModelEthicalSummary
from ml_eval_pro.summary.modules_summary.model_reliability_summary import ModelReliabilitySummary
from ml_eval_pro.summary.modules_summary.model_robustness_summary import ModelRobustnessSummary
from ml_eval_pro.summary.modules_summary.model_transparency_summary import ModelTransparencySummary


class GDPREvaluator(BaseEvaluator):
    """
    A class of the GDPR evaluator.
    """

    def __init__(self, features_description, dataset_context,
                 robustness, unethical_features,
                 evaluator: InterfaceEvaluator):
        """
        Initializing the feature importance evaluator.
        :param features_description: a dictionary of the features descriptions.
        :param dataset_context: a string of the context of the dataset.
        param robustness: a flag of the robustness of the model
        param unethical_features: a list of the unethical features and the reasons.
        :param evaluator: an instance of the evaluator used to initialize the main parameters and evaluate it.
        """
        super().__init__(evaluator)

        self.features_description = features_description
        self.dataset_context = dataset_context

        self.robustness = robustness
        self.unethical_features = unethical_features

        self.__model_ethical = None
        self.__model_reliability = None
        self.__model_robustness = None
        self.__model_transparency = None

    def evaluate(self, **kwargs):
        """
        Evaluate the GDPR compliance from the different evaluation analysis provided.
        :param kwargs: the keys needed to evaluate the GDPR compliance.
        """
        super().evaluate(**kwargs)

        print("Evaluating the model GDPR Compliance ...")
        shap_threshold = kwargs["shap_threshold"]
        ece_threshold = kwargs["ece_threshold"]
        llama_model = kwargs['llama_model']

        model_ethical = ModelEthical(features_description=self.features_description,
                                     dataset_context=self.dataset_context,
                                     X_test=self.test_dataset,
                                     unethical_features=self.unethical_features,
                                     llama_model=llama_model)

        model_reliability = ModelReliability(model=self.model_pipeline,
                                             X_test=self.test_dataset,
                                             y_test=self.test_target,
                                             problem_type=self.problem_type,
                                             num_of_classes=self.num_classes)

        model_robustness = ModelRobustness(model=self.model_pipeline,
                                           X_test=self.test_dataset,
                                           y_test=self.test_target,
                                           X_train=self.train_dataset,
                                           y_train=self.train_target,
                                           problem_type=self.problem_type,
                                           robustness=self.robustness)

        model_transparency = ModelTransparency(model=self.model_pipeline,
                                               X_test=self.test_dataset,
                                               y_test=self.test_target,
                                               problem_type=self.problem_type,
                                               shap_threshold=shap_threshold)

        self.__model_ethical = ModelEthicalSummary(model_ethical).get_summary()
        self.__model_reliability = ModelReliabilitySummary(model_reliability, ece_threshold).get_summary()
        self.__model_robustness = ModelRobustnessSummary(model_robustness).get_summary()
        self.__model_transparency = ModelTransparencySummary(model_transparency).get_summary()

    @property
    def model_ethical(self):
        return self.__model_ethical

    @property
    def model_reliability(self):
        return self.__model_reliability

    @property
    def model_robustness(self):
        return self.__model_robustness

    @property
    def model_transparency(self):
        return self.__model_transparency
