from art.attacks.evasion import HopSkipJump

from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.adversarial_attack_substitute import \
    AdversarialAttackSubstitute


class AdversarialAttackHopSkipJump(AdversarialAttackSubstitute):
    """
    A class for generating the HopskipJump adversarial attack by a substitution model.
    """

    def generate_attack_object(self, substitute_model):
        """
        Generate the HopskipJump adversarial attack object given the substitute model.
        :param substitute_model: the generated substitute model given the data of the original model.
        """
        return HopSkipJump(substitute_model)
