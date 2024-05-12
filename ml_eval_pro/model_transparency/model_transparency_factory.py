from ml_eval_pro.model.evaluated_model import EvaluatedModel

from ml_eval_pro.model_transparency.model_transparency import ModelTransparency
from ml_eval_pro.model_transparency.model_transparency_catboost import ModelTransparencyCatBoost
from ml_eval_pro.model_transparency.model_transparency_h2o import ModelTransparencyH2O
from ml_eval_pro.model_transparency.model_transparency_keras import ModelTransparencyKeras
from ml_eval_pro.model_transparency.model_transparency_lightgbm import ModelTransparencyLightGBM
from ml_eval_pro.model_transparency.model_transparency_onnx import ModelTransparencyONNX
from ml_eval_pro.model_transparency.model_transparency_pytorch import ModelTransparencyPyTorch
from ml_eval_pro.model_transparency.model_transparency_sklearn import ModelTransparencySKlearn
from ml_eval_pro.model_transparency.model_transparency_sparkmllib import ModelTransparencySparkML
from ml_eval_pro.model_transparency.model_transparency_statsmodel import ModelTransparencyStatsModel
from ml_eval_pro.model_transparency.model_transparency_tensorflow import ModelTransparencyTensorflow
from ml_eval_pro.model_transparency.model_transparency_xgboost import ModelTransparencyXGBoost


class ModelTransparencyFactory:
    """
    A class for generating a model transparency object.
    """

    @classmethod
    def create(cls, model: EvaluatedModel, *args, **kwargs) -> ModelTransparency:
        """
        Create a model based on the model type.
        :param model: the model object.
        :return: the created model class according to its type.
        """
        _factory_supported_classes = {"mlflow.catboost": ModelTransparencyCatBoost,
                                      "mlflow.h2o": ModelTransparencyH2O,
                                      "mlflow.keras": ModelTransparencyKeras,
                                      "mlflow.lightgbm": ModelTransparencyLightGBM,
                                      "mlflow.onnx": ModelTransparencyONNX,
                                      "mlflow.pytorch": ModelTransparencyPyTorch,
                                      "mlflow.sklearn": ModelTransparencySKlearn,
                                      "mlflow.spark": ModelTransparencySparkML,
                                      "mlflow.statsmodel": ModelTransparencyStatsModel,
                                      "mlflow.tensorflow": ModelTransparencyTensorflow,
                                      "mlflow.xgboost": ModelTransparencyXGBoost
                                      }

        if model.model_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(model.model_type)
            return subclass(model, *args, **kwargs)
        else:
            raise NotImplementedError(f"The model type {model.model_type} is not supported!")