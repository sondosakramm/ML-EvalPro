from typing import Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor, \
    AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, StackingClassifier, \
    StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, \
    RadiusNeighborsRegressor
from sklearn.svm import SVC, NuSVC, SVR, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

from ml_eval_pro.transparency.transparency import Transparency


class TransparencySKlearn(Transparency):
    """
    A class for generating transparency for sklearn models.
    """

    def get_model_algorithm(self):
        return self.model.model

    def get_model_algorithms_complexity(self) -> Tuple[list, list, list]:
        explainable_models = [LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, DecisionTreeClassifier,
                              DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor]

        partially_explainable_models = [KNeighborsClassifier, KNeighborsRegressor,
                                        RadiusNeighborsClassifier, RadiusNeighborsRegressor, SVC, NuSVC,
                                        SVR, NuSVR]

        complex_models = [RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor,
                          AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
                          StackingClassifier, StackingRegressor]

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
