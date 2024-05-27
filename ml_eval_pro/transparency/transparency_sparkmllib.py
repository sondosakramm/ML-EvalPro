from typing import Tuple

from pyspark.ml.classification import NaiveBayesModel, LogisticRegressionModel, LinearSVCModel,  \
    DecisionTreeClassifier, GBTClassifier, MultilayerPerceptronClassifier, FMClassifier, RandomForestClassifier
from pyspark.ml.regression import DecisionTreeRegressor, GBTRegressor, \
    GeneralizedLinearRegression, IsotonicRegression, LinearRegression, RandomForestRegressor
from pyspark.mllib.classification import SVMWithSGD, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD
from pyspark.mllib.regression import IsotonicRegression, LassoWithSGD, RidgeRegressionWithSGD, \
    LinearRegressionWithSGD, LinearModel
from pyspark.mllib.tree import GradientBoostedTreesModel, RandomForestModel, DecisionTreeModel

from ml_eval_pro.transparency.transparency import Transparency


class TransparencySparkML(Transparency):
    """
    A class for generating transparency for sparkml models.
    """


    def get_model_algorithm(self):
        return self.model.model.stages[0]

    def get_model_algorithms_complexity(self) -> Tuple[list, list, list]:
        explainable_models = [LinearRegressionWithSGD, LinearModel,
                              GeneralizedLinearRegression, LinearRegression,
                              LogisticRegressionModel, LogisticRegressionWithLBFGS, LogisticRegressionWithSGD,
                              IsotonicRegression, RidgeRegressionWithSGD, LassoWithSGD,
                              DecisionTreeModel, DecisionTreeClassifier, DecisionTreeRegressor,
                              NaiveBayesModel]

        partially_explainable_models = [LinearSVCModel, SVMWithSGD]

        complex_models = [RandomForestModel, RandomForestClassifier, RandomForestRegressor,
                          GradientBoostedTreesModel, GBTClassifier, GBTRegressor,
                          MultilayerPerceptronClassifier,
                          FMClassifier]

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
