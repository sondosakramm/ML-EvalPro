import mlflow

from ml_eval_pro.model.evaluated_model import EvaluatedModel
from ml_eval_pro.model.evaluated_model_h2o import EvaluatedModelH2O
from ml_eval_pro.model.evaluated_model_pytorch import EvaluatedModelPytorch
from ml_eval_pro.model.evaluated_model_sklearn import EvaluatedModelSKLearn
from ml_eval_pro.model.evaluated_model_tensorflow import EvaluatedModelTensorflow


class EvaluatedModelFactory:
    """
    A class for generating a model object.
    """

    @classmethod
    def create(cls, model_uri: str, *args, **kwargs) -> EvaluatedModel:
        """
        Create a model based on the model type.
        :param model_uri: the model uri.
        :return: the created model class according to its type.
        """
        model_info = mlflow.models.get_model_info(model_uri)
        model_type = model_info.flavors[mlflow.pyfunc.FLAVOR_NAME]["loader_module"]

        _factory_supported_classes = {"mlflow.sklearn": EvaluatedModelSKLearn,
                                      # "mlflow.pytorch": EvaluatedModel,
                                      "mlflow.h2o": EvaluatedModelH2O,
                                      # "mlflow.sparkmllib": EvaluatedModel
                                      }

        print(f"Constructing the model {model_type} ...")
        if model_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(model_type)
            return subclass(model_uri, *args, **kwargs)
        else:
            return EvaluatedModel(model_uri, *args, **kwargs)
