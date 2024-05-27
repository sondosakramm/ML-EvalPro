from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.model.adversarial_attack_substitute_model import \
    AdversarialAttackSubstituteModel


class AdversarialAttackSubstituteClassification(AdversarialAttackSubstituteModel):
    def generate_predictions(self):
        """
        Generate the predictions used for training the substitute model.
        """
        return self.train_model_predictions
