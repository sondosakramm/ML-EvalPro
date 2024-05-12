import pandas as pd

from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute import AdversarialAttackSubstitute
from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance


class ModelRobustness(GdprCompliance):

    def get_evaluation(self):
        """
        Get adversarial attacks and their evaluation.
        """
        if self.adversarial_attack_model is None:
            self.adversarial_attacks = AdversarialAttackSubstitute(self.model, self.problem_type,
                                                                   self.X_test, self.y_test, self.X_train, self.y_train)
            self.attacks = self.adversarial_attacks.generate()
            self.attacks_df = pd.DataFrame(self.attacks, columns=self.X_test.columns)
            return self.adversarial_attacks.evaluate_robustness(self.model.predict(self.attacks_df))

        return self.adversarial_attack_model.evaluate_robustness(self.model.predict(self.adversarial_testcases))
