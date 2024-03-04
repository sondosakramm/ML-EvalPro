from ml_eval_pro.gdpr.gdpr_rules.model_ethical import ModelEthical
from ml_eval_pro.summary.summary_generator import SummaryGenerator


class ModelEthicalSummary(SummaryGenerator):
    def __init__(self, model_ethical: ModelEthical):
        self.model_ethical = model_ethical

    def get_summary(self):
        summary_str = f'{5 * "*"}\tModel Ethical\t{5 * "*"}\n'
        feature_unethical = self.model_ethical.get_unethical_features()
        for i in range(len(feature_unethical['feature'])):
            summary_str += (f'Feature {feature_unethical["feature"][i]} is not ethical because '
                            f'{feature_unethical["reason"][i]}\n')
        return summary_str
