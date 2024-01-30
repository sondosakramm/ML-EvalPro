from auto_evaluator.adverserial_test_cases.adversarial_attack_substitute import AdversarialAttackSubstitute
from auto_evaluator.gdpr.gdpr_compliance import GdprCompliance


class ModelRobustness(GdprCompliance):


    def __get_evaluation(self):
        """
        Get adversarial attacks and their evaluation.
        """
        self.adversarial_attacks = AdversarialAttackSubstitute(self.model, self.problem_type, self.X_test, self.y_test)
        self.attacks = self.adversarial_attacks.generate()
        return self.adversarial_attacks.evaluate_robustness(self.attacks)


    def __str__(self):
        summary = f'{5 * "*"}\tModel Security\t{5 * "*"}\n'
        if self.__get_evaluation():
            summary += (f'Model is Robust.\nPredictions of the original dataset are:\n'
                        f'{self.attacks}'
                        f'\nPredictions of the Adversarial attacks are:\n'
                        f'{self.X_test}')
        else:
            summary += (f'Model is NOT Robust.\nPredictions of the original dataset are:\n'
                        f'{self.attacks}'
                        f'\nPredictions of the Adversarial attacks are:\n'
                        f'{self.X_test}')
        return summary
