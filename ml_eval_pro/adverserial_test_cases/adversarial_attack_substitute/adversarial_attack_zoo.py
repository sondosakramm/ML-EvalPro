from art.attacks.evasion.zoo import ZooAttack

from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.adversarial_attack_substitute import \
    AdversarialAttackSubstitute


class AdversarialAttackZOO(AdversarialAttackSubstitute):
    """
    A class for generating the ZOO attack by a substitution model.
    """

    def generate_attack_object(self, substitute_model):
        """
        Generate the ZOO adversarial attack object given the substitute model.
        :param substitute_model: the generated substitute model given the data of the original model.
        """
        return ZooAttack(substitute_model, batch_size=1, nb_parallel=1)
