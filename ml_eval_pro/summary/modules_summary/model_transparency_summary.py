from ml_eval_pro.gdpr.gdpr_rules.model_transparency import ModelTransparency
from ml_eval_pro.summary.summary_generator import SummaryGenerator


class ModelTransparencySummary(SummaryGenerator):
    def __init__(self, model_trans: ModelTransparency):
        self.model_trans = model_trans

    def __get_explain_ability_summary(self):
        explain_summary = f''
        exp = self.model_trans.check_explain_ability()
        if exp == "A":
            explain_summary += ('The model is explainable because it falls into a category of models known for '
                                'their interpretability.')
        elif exp == "B":
            explain_summary += ('The model is partially explainable. '
                                'While some aspects are interpretable, it may have complex components.')
        elif exp == "C":
            explain_summary += f'The model is complex and less explainable.'
        elif exp == "I":
            explain_summary += f'Model type is not supported.'
        return explain_summary

    def __get_interpretability_summary(self):
        inter_summary = f''
        if self.model_trans.check_significance():
            inter_summary += (
                f'The average entropy value for the distributions of the instance-specific feature importance is '
                f'{self.model_trans.avg_entropy:.4f}.\n'
                f'Low entropy signifies that the model predictions are driven by clear and '
                f'distinguishable patterns within the input features.\n')
        else:
            inter_summary += ('The models interpretability may be more challenging. '
                              'Which indicates a greater degree of complexity and potential interactions among '
                              'features in influencing predictions.\n')
        return inter_summary

    def get_summary(self):
        return (f'{5 * "*"}\tModel Transparency\t{5 * "*"}\n' + self.__get_explain_ability_summary()
                + self.__get_interpretability_summary())
