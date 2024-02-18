from ml_eval_pro.ethical_analysis.feature_importance.feature_importance import FeatureImportance
from ml_eval_pro.ethical_analysis.feature_importance.shap import SHAP

class FeatureImportanceFactory:
    """
    A class for generating a feature importance object.
    """

    @classmethod
    def create(cls, feature_importance_type: str, *args, **kwargs) -> FeatureImportance:
        """
        Create a feature importance based on the type.
        :param feature_importance_type: the input feature importance type.
        :return: the created feature importance class according to its type.
        """
        _factory_supported_classes = {"shap": SHAP}

        if feature_importance_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(feature_importance_type)
            return subclass(*args ,**kwargs)
        else:
            raise Exception(f'Cannot find "{feature_importance_type}"')