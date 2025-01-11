from ml_eval_pro.gdpr.gdpr_rules.model_reliability import ModelReliability
from ml_eval_pro.summary.summary_generator import SummaryGenerator


class ModelReliabilitySummary(SummaryGenerator):
    def __init__(self, model_reliab: ModelReliability, ece_threshold: float = 0.5):
        self.model_reliab = model_reliab
        self.ece_threshold = ece_threshold

    def get_summary(self):
        summary_str = f'{5 * "*"}\tModel Reliability\t{5 * "*"}\n'
        if self.model_reliab.problem_type == 'classification':
            if self.model_reliab.get_metric() > self.ece_threshold:
                summary_str += (
                    f'The model reliability evaluation showed a mismatch between the model confidence level '
                    f'and the accuracy of its predictions by '
                    f'{self.model_reliab.get_metric() * 100: .5f}% on average.\n')
            else:
                summary_str = (
                    f'The model reliability evaluation indicates that the model is reliable.'
                )

        elif self.model_reliab.problem_type == 'regression':
            summary_str = (
                f'The model reliability evaluation can be observed from the graph.'
            )
        else:
            summary_str += f'Problem type is not supported.'
        return summary_str
