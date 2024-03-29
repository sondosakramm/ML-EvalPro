from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute import AdversarialAttackSubstitute
from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance


class ModelRobustness(GdprCompliance):

    def get_evaluation(self):
        """
        Get adversarial attacks and their evaluation.
        """
        self.adversarial_attacks = AdversarialAttackSubstitute(self.model, self.problem_type, self.X_test, self.y_test)
        self.attacks = self.adversarial_attacks.generate()
        return self.adversarial_attacks.evaluate_robustness(self.model.predict(self.attacks))
