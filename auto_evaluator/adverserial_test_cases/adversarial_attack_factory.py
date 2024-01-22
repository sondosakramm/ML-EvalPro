from auto_evaluator.adverserial_test_cases.adversarial_attack import AdversarialAttack
from auto_evaluator.adverserial_test_cases.adversarial_evasion_attack import AdversarialEvasionAttack


class AdversarialAttackFactory:
    """
    A class for generating an adversarial attack object.
    """

    @classmethod
    def create(cls, attack_type: str, *args, **kwargs) -> AdversarialAttack:
        """
        Create an adversarial attack based on the attack type.
        :param attack_type: the attack type.
        :return: the created attack class according to its type.
        """
        _factory_supported_classes = {"evasion_attack": AdversarialEvasionAttack}

        if attack_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(attack_type)
            return subclass(*args ,**kwargs)
        else:
            raise Exception(f'Cannot find "{attack_type}"')