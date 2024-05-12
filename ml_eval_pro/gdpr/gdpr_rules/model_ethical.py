import string

from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance
from ml_eval_pro.ethical_analysis.ethical_analysis import EthicalAnalysis


class ModelEthical(GdprCompliance):
    def get_unethical_features(self):
        if self.unethical_features is None:
            return EthicalAnalysis.prompt_feature_ethnicity(self.features_description, self.dataset_context,
                                                            self.X_test)

        return self.unethical_features
