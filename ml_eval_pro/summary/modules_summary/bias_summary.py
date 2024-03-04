import numpy as np

from ml_eval_pro.bias.model_bias import ModelBias
from ml_eval_pro.summary.summary_generator import SummaryGenerator


class BiasSummary(SummaryGenerator):
    def __init__(self, model_bias: ModelBias):
        self.model_bias = model_bias

    def get_summary(self):
        if len(self.model_bias.features_bias) == 0:
            return "There is no bias features in this model."

        features_bias_np = np.array(self.model_bias.features_bias)

        ', '.join(map(str, features_bias_np[:, 0]))
        features_names = ', '.join(map(str, features_bias_np[:, 0]))
        features_values = ', '.join(map(str, features_bias_np[:, 1]))

        return (f"By calculating the predictions of different possible values of each feature,"
                f" the biased features are {features_names} The corresponding bias statistics are {features_values}"
                f" respectively, given that values closer to 0 indicate less bias.")
