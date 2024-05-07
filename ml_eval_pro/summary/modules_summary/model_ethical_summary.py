from ml_eval_pro.gdpr.gdpr_rules.model_ethical import ModelEthical
from ml_eval_pro.summary.summary_generator import SummaryGenerator


class ModelEthicalSummary(SummaryGenerator):
    def __init__(self, model_ethical: ModelEthical):
        self.model_ethical = model_ethical

    def get_summary(self):
        summary_str = f'{5 * "*"}\tModel Ethical\t{5 * "*"}\n'
        summary_str += self.model_ethical.get_unethical_features()
        return summary_str
