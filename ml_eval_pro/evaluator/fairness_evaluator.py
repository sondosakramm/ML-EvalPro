from ml_eval_pro.evaluator.base_evaluator import BaseEvaluator
from ml_eval_pro.evaluator.interface_evaluator import InterfaceEvaluator
from ml_eval_pro.model_fairness.model_fairness_factory import ModelFairnessFactory
from ml_eval_pro.summary.modules_summary.bias_summary import BiasSummary
from ml_eval_pro.summary.modules_summary.equalized_odds_summary import EqualizedOddsSummary


class FairnessEvaluator(BaseEvaluator):
    """
    A class of the fairness evaluator.
    """

    def __init__(self, evaluator: InterfaceEvaluator):
        """
        Initializing the fairness evaluator.
        :param evaluator: an instance of the evaluator used to initialize the main parameters and evaluate it.
        """
        super().__init__(evaluator)

        self.__biased_features = None
        self.__bias_summary = None
        self.__equalized_odds = None
        self.__equalized_odds_summary = None

    def evaluate(self, **kwargs):
        """
        Evaluate the bias from the different evaluation analysis provided.
        :param kwargs: the keys needed to evaluate the bias.
        """
        super().evaluate(**kwargs)

        print("Evaluating the model bias and fairness...")
        eval_metric = 'MAPE' if self.problem_type == 'regression' else 'Accuracy'

        threshold = kwargs['bias_threshold']

        model_bias = ModelFairnessFactory.create("bias",
                                                 model=self.model_pipeline,
                                                 model_type=self.problem_type,
                                                 data=self.test_dataset,
                                                 target=self.test_target,
                                                 evaluation_metrics=[eval_metric],
                                                 threshold=threshold)

        biased_features_dict = model_bias.get_model_fairness()
        self.__biased_features = list(biased_features_dict.keys())
        self.__bias_summary = BiasSummary(biased_features_dict, threshold).get_summary()

        if self.problem_type == 'regression':
            self.__equalized_odds = {}
            self.__equalized_odds_summary = ""
        else:
            model_equalized_odds = ModelFairnessFactory.create("equalized odds",
                                                               model=self.model_pipeline,
                                                               model_type=self.problem_type,
                                                               data=self.test_dataset,
                                                               target=self.test_target)

            self.__equalized_odds = model_equalized_odds.get_model_fairness()
            self.__equalized_odds_summary = EqualizedOddsSummary(self.__equalized_odds).get_summary()

    @property
    def biased_features(self):
        return self.__biased_features

    @property
    def equalized_odds(self):
        return self.__equalized_odds

    @property
    def bias_summary(self):
        return self.__bias_summary

    @property
    def equalized_odds_summary(self):
        return self.__equalized_odds_summary
