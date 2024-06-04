from ml_eval_pro.evaluator.base_evaluator import BaseEvaluator
from ml_eval_pro.evaluator.interface_evaluator import InterfaceEvaluator
from ml_eval_pro.summary.modules_summary.variance_summary import VarianceSummary
from ml_eval_pro.variance.model_var.model_variance_factory import ModelVarianceFactory


class VarianceEvaluator(BaseEvaluator):
    """
    A class of the variance evaluator.
    """

    def __init__(self, evaluator: InterfaceEvaluator):
        """
        Initializing the variance evaluator.
        :param evaluator: an instance of the evaluator used to initialize the main parameters and evaluate it.
        """
        super().__init__(evaluator)

        self.__train_variance_value = None
        self.__high_variance_features = None
        self.__variance_summary = None

    def evaluate(self, **kwargs):
        """
        Evaluate the variance from the different evaluation analysis provided.
        :param kwargs: the keys needed to evaluate the variance.
        """
        super().evaluate(**kwargs)

        print("Evaluating the model variance ...")
        eval_metric = 'MAE' if self.problem_type == 'regression' else 'Accuracy'

        variance_threshold = kwargs['variance_threshold']
        variance_step_size = kwargs['variance_step_size']

        if self.train_target is not None:
            model_variance = ModelVarianceFactory.create(variance_type='train_test_data',
                                                         model=self.model_pipeline,
                                                         model_type=self.problem_type,
                                                         train_dataset=self.train_dataset,
                                                         train_target=self.train_target,
                                                         test_dataset=self.test_dataset,
                                                         target=self.test_target,
                                                         evaluation_metric=eval_metric,
                                                         threshold=variance_threshold)

            self.__train_variance_value = model_variance.calculate_variance()
            self.__variance_summary = VarianceSummary(model_variance).get_summary() + "\n\n"

        model_variance = ModelVarianceFactory.create(variance_type='test_data',
                                                     model=self.model_pipeline,
                                                     model_type=self.problem_type,
                                                     test_dataset=self.test_dataset,
                                                     target=self.test_target,
                                                     evaluation_metric=eval_metric,
                                                     threshold=variance_threshold,
                                                     step_size=variance_step_size)
        model_variance.calculate_variance()
        self.__high_variance_features = model_variance.get_diff()

        if self.__variance_summary is None:
            self.__variance_summary = ""

        self.__variance_summary += VarianceSummary(model_variance).get_summary()

    @property
    def train_variance_value(self):
        return self.__train_variance_value

    @property
    def high_variance_features(self):
        return self.__high_variance_features

    @property
    def variance_summary(self):
        return self.__variance_summary
