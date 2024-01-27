from auto_evaluator.gdpr.gdpr_compliance import GdprCompliance


class ModelEthical(GdprCompliance):

    def __get_unethical_features(self):
        # Use LLms and get unethical features only.
        pass

    def __str__(self):
        summary_str = f'{5 * "*"} Model Ethical {5 * "*"}\n'

        return summary_str
