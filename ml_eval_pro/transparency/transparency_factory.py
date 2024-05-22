from ml_eval_pro.model.evaluated_model import EvaluatedModel

from ml_eval_pro.transparency.transparency import Transparency
from ml_eval_pro.transparency.transparency_catboost import TransparencyCatBoost
from ml_eval_pro.transparency.transparency_h2o import TransparencyH2O
from ml_eval_pro.transparency.transparency_lightgbm import TransparencyLightGBM
from ml_eval_pro.transparency.transparency_onnx import TransparencyONNX
from ml_eval_pro.transparency.transparency_pytorch import TransparencyPyTorch
from ml_eval_pro.transparency.transparency_sklearn import TransparencySKlearn
from ml_eval_pro.transparency.transparency_sparkmllib import TransparencySparkML
from ml_eval_pro.transparency.transparency_statsmodel import ModelTransparencyStats
from ml_eval_pro.transparency.transparency_tensorflow import TransparencyTensorflow
from ml_eval_pro.transparency.transparency_xgboost import TransparencyXGBoost


class TransparencyFactory:
    """
    A class for generating a model transparency object.
    """

    @classmethod
    def create(cls, model: EvaluatedModel, *args, **kwargs) -> Transparency:
        """
         Create a model based on the model type.
        :param model: the model object.
        :return: the created model class according to its type.
        """
        _factory_supported_classes = {"mlflow.catboost": TransparencyCatBoost,
                                      "mlflow.h2o": TransparencyH2O,
                                      "mlflow.lightgbm": TransparencyLightGBM,
                                      "mlflow.onnx": TransparencyONNX,
                                      "mlflow.pytorch": TransparencyPyTorch,
                                      "mlflow.sklearn": TransparencySKlearn,
                                      "mlflow.spark": TransparencySparkML,
                                      "mlflow.statsmodels": ModelTransparencyStats,
                                      "mlflow.tensorflow": TransparencyTensorflow,
                                      "mlflow.xgboost": TransparencyXGBoost
                                      }

        if model.model_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(model.model_type)
            return subclass(model)
        else:
            raise NotImplementedError(f"The model type {model.model_type} is not supported!")
