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
    def create(cls, model_type: str, *args, **kwargs) -> EvaluatedModel:
        """
        Create a model based on the model type.
        :param model_type: the model type.
        :return: the created model class according to its type.
        """
        _factory_supported_classes = {"sklearn": EvaluatedModelSKLearn,
                                      "pytorch": EvaluatedModelPytorch,
                                      "h2o": EvaluatedModelH2O,
                                      "tensorflow": EvaluatedModelTensorflow}

        if model_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(model_type)
            return subclass(*args, **kwargs)
        else:
            raise Exception(f'Cannot find "{model_type}"')
