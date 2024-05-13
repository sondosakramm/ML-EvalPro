import numpy as np

from ml_eval_pro.utils.optimal_clusters import calculate_optimal_bins
from ml_eval_pro.adverserial_test_cases.adversarial_attack_substitute.model.adversarial_attack_substitute_model import \
    AdversarialAttackSubstituteModel


class AdversarialAttackSubstituteRegression(AdversarialAttackSubstituteModel):

    def generate_predictions(self):
        num_bins = calculate_optimal_bins(self.train_model_predictions)
        binning_ranges = np.linspace(min(self.train_model_predictions), max(self.train_model_predictions) + 1,
                                     num_bins + 1)
        training_labels_true = np.digitize(self.train_model_predictions, bins=binning_ranges)

        return training_labels_true
