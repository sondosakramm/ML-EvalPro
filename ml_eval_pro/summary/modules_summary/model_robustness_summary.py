from ml_eval_pro.gdpr.gdpr_rules.model_robustness import ModelRobustness
from ml_eval_pro.summary.summary_generator import SummaryGenerator


class ModelRobustnessSummary(SummaryGenerator):
    def __init__(self, model_robs: ModelRobustness):
        self.model_robs = model_robs

    def __get_robustness(self):
        if self.model_robs.get_evaluation():
            return f'Model is Robust, No adversarial attacks found on the dataset!'
        else:
            return f'Model is NOT Robust. There exists some adversarial attacks found on the dataset!'

    def get_summary(self):
        return f'{5 * "*"}\tModel Robustness\t{5 * "*"}\n' + self.__get_robustness()
