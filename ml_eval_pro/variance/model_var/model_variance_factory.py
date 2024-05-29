from ml_eval_pro.variance.model_var.model_variance import ModelVariance
from ml_eval_pro.variance.model_variance_by_test_data import ModelVarianceByTestData
from ml_eval_pro.variance.model_variance_by_train_test_data import ModelVarianceByTrainTestData


class ModelVarianceFactory:
    """
    A class for generating a model variance object.
    """

    @classmethod
    def create(cls, variance_type: str, *args, **kwargs) -> ModelVariance:
        """
        Create a variance based on the type.
        :param variance_type: the input variance type.
        :return: the created variance class according to its type.
        """
        _factory_supported_classes = {"train_test_data": ModelVarianceByTrainTestData,
                                      "test_data": ModelVarianceByTestData}

        if variance_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(variance_type)
            return subclass(*args, **kwargs)
        else:
            raise Exception(f'Cannot find "{variance_type}"')
