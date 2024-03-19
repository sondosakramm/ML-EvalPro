import string

from ml_eval_pro.gdpr.gdpr_compliance import GdprCompliance
from ml_eval_pro.ethical_analysis.ethical_analysis import EthicalAnalysis

class ModelEthical(GdprCompliance):
    def get_unethical_features(self):
        if self.features_description is None:
            return "Unable to address ethical concerns at this time, as no description or details have been provided."
        else:
            return EthicalAnalysis.prompt_feature_ethnicity(self.features_description)

