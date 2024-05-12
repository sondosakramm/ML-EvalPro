from typing import Tuple

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor, \
    AdaBoostClassifier, AdaBoostRegressor, GradientBoostingClassifier, GradientBoostingRegressor, StackingClassifier, \
    StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, ElasticNet, Lasso, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, \
    RadiusNeighborsRegressor
from sklearn.svm import SVC, NuSVC, SVR, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

from ml_eval_pro.model_transparency.model_transparency import ModelTransparency


class ModelTransparencySKlearn(ModelTransparency):
    def get_model_algorithm(self):
        return self.model.__dict__["model"]

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
