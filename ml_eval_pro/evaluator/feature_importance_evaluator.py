from ml_eval_pro.ethical_analysis.ethical_analysis import EthicalAnalysis
from ml_eval_pro.evaluator.base_evaluator import BaseEvaluator
from ml_eval_pro.evaluator.interface_evaluator import InterfaceEvaluator


class FeatureImportanceEvaluator(BaseEvaluator):
    """
    A class of the feature importance evaluator.
    """
    def __init__(self, features_description, dataset_context,
                 evaluator: InterfaceEvaluator):
        """
        Initializing the feature importance evaluator.
        :param features_description: a dictionary of the features descriptions.
        :param dataset_context: a string of the context of the dataset.
        :param evaluator: an instance of the evaluator used to initialize the main parameters and evaluate it.
        """
        super().__init__(evaluator)

        self.features_description = features_description
        self.dataset_context = dataset_context

        self.__features_importance_scores = None
        self.__unethical_features = None

    def evaluate(self, **kwargs):
        """
        Evaluate the ethical perspective and feature importance from the different evaluation analysis provided.
        :param kwargs: the keys needed to evaluate the ethical perspective and feature importance.
        """
        super().evaluate(**kwargs)

        print("Evaluating the model ethical issues according to the features importance ...")
        llama_model = kwargs['llama_model']
        ethical_analysis = EthicalAnalysis(self.model_pipeline, self.test_dataset,
                                           self.features_description, self.dataset_context)
        self.__features_importance_scores, self.__unethical_features = ethical_analysis(llama_model=llama_model)

    @property
    def features_importance_scores(self):
        return self.__features_importance_scores

    @property
    def unethical_features(self):
        return self.__unethical_features
