from typing import Tuple

from h2o.estimators import H2ODeepLearningEstimator, H2OGradientBoostingEstimator, H2ORandomForestEstimator, \
    H2OGeneralizedLinearEstimator, H2ODecisionTreeEstimator, H2OGeneralizedAdditiveEstimator, \
    H2ONaiveBayesEstimator, H2OSupportVectorMachineEstimator, H2OUpliftRandomForestEstimator, H2OAdaBoostEstimator, \
    H2OStackedEnsembleEstimator, H2OXGBoostEstimator

from ml_eval_pro.transparency.transparency import Transparency


class TransparencyH2O(Transparency):
    """
    A class for generating transparency for h2o models.
    """

    def get_model_algorithm(self):
        return self.model.model._model_impl.h2o_model

    def get_model_algorithms_complexity(self) -> Tuple[list, list, list]:
        explainable_models = [H2OGeneralizedLinearEstimator, H2ODecisionTreeEstimator,
                              H2OGeneralizedAdditiveEstimator, H2ONaiveBayesEstimator]

        partially_explainable_models = [H2OSupportVectorMachineEstimator]

        complex_models = [H2ODeepLearningEstimator, H2OStackedEnsembleEstimator,
                          H2OAdaBoostEstimator, H2OGradientBoostingEstimator,
                          H2ORandomForestEstimator, H2OUpliftRandomForestEstimator, H2OXGBoostEstimator]

        return explainable_models, partially_explainable_models, complex_models

    def get_model_score(self, model_algorithm, **kwargs):
        explainable_models = kwargs["explainable_models"]
        partially_explainable_models = kwargs["partially_explainable_models"]
        complex_models = kwargs["complex_models"]

        if any(isinstance(model_algorithm, model_type) for model_type in explainable_models):
            return 25
        elif any(isinstance(model_algorithm, model_type) for model_type in partially_explainable_models):
            return 65
        elif any(isinstance(model_algorithm, model_type) for model_type in complex_models):
            return 110
        else:
            return -1
