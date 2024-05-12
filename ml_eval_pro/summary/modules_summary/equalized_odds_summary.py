from ml_eval_pro.summary.summary_generator import SummaryGenerator


class EqualizedOddsSummary(SummaryGenerator):
    def __init__(self, equalized_odds_vals: dict):
        self.equalized_odds_vals = equalized_odds_vals

    def get_summary(self):
        res_str = (f"Equalized odds refers to the idea that the predictions of a model"
                   f" should have equal false positive rates (FPR) and equal true positive rates (TPR)"
                   f" across different sensitive groups defined by a protected attribute."
                   f" This metric aims to ensure fairness and prevent discrimination in algorithmic decision-making."
                   f"\nIn this dataset, the equalized odds value over ")

        equalized_odds_features_names = self.equalized_odds_vals.keys()
        equalized_odds_features_values = self.equalized_odds_vals.values()

        features_names = ', '.join(map(str, equalized_odds_features_names))
        features_values = ', '.join(map(str, equalized_odds_features_values))

        res_str += f"{features_names} are {features_values} respectively."

        return res_str
