from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.model.adversarial_attack_substitute_classification import \
    AdversarialAttackSubstituteClassification
from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.model.adversarial_attack_substitute_model import \
    AdversarialAttackSubstituteModel
from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.model.adversarial_attack_substitute_regression import \
    AdversarialAttackSubstituteRegression


class AdversarialAttackSubstituteModelFactory:
    """
    A class for generating an adversarial attack substitute model object.
    """
    @classmethod
    def create(cls, model_type: str, *args, **kwargs) -> AdversarialAttackSubstituteModel:
        """
        Create an adversarial attack substitute model based on the attack type.
        :param model_type: the model type (regression or classification).
        :return: the created adversarial attack substitute model class according to its type.
        """
        _factory_supported_classes = {"regression": AdversarialAttackSubstituteRegression,
                                      "classification": AdversarialAttackSubstituteClassification}

        if model_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(model_type)
            return subclass(*args, **kwargs)
        else:
            raise Exception(f'Cannot find "{model_type}"')
