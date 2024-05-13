from ml_eval_pro.adverserial_test_cases.adversarial_attack import AdversarialAttack
from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.adversarial_attack_zoo\
    import AdversarialAttackZOO
from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.adversrial_attack_hopskipjump\
    import AdversarialAttackHopSkipJump


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
        _factory_supported_classes = {"substitute_zoo": AdversarialAttackZOO,
                                      "substitute_hopskipjump": AdversarialAttackHopSkipJump}

        if attack_type in _factory_supported_classes:
            subclass = _factory_supported_classes.get(attack_type)
            return subclass(*args, **kwargs)
        else:
            raise Exception(f'Cannot find "{attack_type}"')