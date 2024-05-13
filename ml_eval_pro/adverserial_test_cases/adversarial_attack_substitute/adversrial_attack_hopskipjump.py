from art.attacks.evasion import HopSkipJump

from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.adversarial_attack_substitute import \
    AdversarialAttackSubstitute


class AdversarialAttackHopSkipJump(AdversarialAttackSubstitute):
    """
    A class for generating an adversarial attack by a substitution model.
    """

    def generate_attack_object(self, substitute_model):
        return HopSkipJump(substitute_model)
