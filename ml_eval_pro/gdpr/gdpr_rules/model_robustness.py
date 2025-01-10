from ml_eval_pro.adverserial_test_cases.adversarial_attack_factory import AdversarialAttackFactory
from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance


class ModelRobustness(GdprCompliance):

    def get_evaluation(self):
        """
        Get adversarial attacks and their evaluation.
        """
        if self.robustness is None:
            dataset_columns = list(self.X_test.columns)

            adversarial_attack_model = AdversarialAttackFactory.create("substitute_zoo",
                                                                       self.model,
                                                                       self.problem_type,
                                                                       self.X_test, self.y_test, dataset_columns,
                                                                       train_input_features=self.X_train,
                                                                       train_target_features=self.y_train,
                                                                       num_classes=self.num_of_classes)

            _ = adversarial_attack_model.get_adversarial_testcases()
            return adversarial_attack_model.is_robust

        return self.robustness
