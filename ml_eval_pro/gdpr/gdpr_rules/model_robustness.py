from ml_eval_pro.adverserial_test_cases.adversarial_attack_factory import AdversarialAttackFactory
from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance


class ModelRobustness(GdprCompliance):

    def get_evaluation(self):
        """
        Get adversarial attacks and their evaluation.
        """
        if self.adversarial_attack_model is None:
            dataset_columns = list(self.X_test.columns)

            self.adversarial_attack_model = (
                AdversarialAttackFactory.create("substitute_hopskipjump",
                                                self.model,
                                                self.problem_type,
                                                self.X_test, self.y_test, dataset_columns,
                                                num_classes=self.num_of_classes) if self.y_train is None else

                AdversarialAttackFactory.create("substitute_hopskipjump",
                                                self.model,
                                                self.problem_type,
                                                self.X_test, self.y_test, dataset_columns,
                                                train_input_features=self.X_train,
                                                train_target_features=self.y_train,
                                                num_classes=self.num_of_classes))
            adversarial_testcases = self.adversarial_attack_model.get_adversarial_testcases()

        return self.adversarial_attack_model.is_robust
