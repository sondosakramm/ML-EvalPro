import os

import numpy as np

from ml_eval_pro.bias.model_bias import ModelBias
from ml_eval_pro.configuration_manager.configuration_reader.yaml_reader import YamlReader
from ml_eval_pro.summary.summary_generator import SummaryGenerator


class BiasSummary(SummaryGenerator):
    def __init__(self, model_bias: ModelBias):
        self.model_bias = model_bias
        yaml_reader = YamlReader(os.path.join(os.path.curdir, "ml_eval_pro",
                                              "config_files", "system_config.yaml"))
        self.threshold = yaml_reader.get('thresholds')['bias_threshold']

    def get_summary(self):
        res_str = (f"By dividing the data by the unique values of each feature, each feature is evaluated independently,"
                   f" and the model is evaluated for each unique value. Next, by comparing the differences"
                   f" in each value, the feature's absolute average performance is calculated.\n"
                   f"A comparison of the absolute average performances ")

        if len(self.model_bias.features_bias) == 0:
            res_str += "indicates that there is no bias in any of the features."
            return res_str

        features_bias_np = np.array(self.model_bias.features_bias)

        ', '.join(map(str, features_bias_np[:, 0]))
        features_names = ', '.join(map(str, features_bias_np[:, 0]))
        features_values = ', '.join(map(str, features_bias_np[:, 1]))

        res_str += (f"indicates a bias in the features {features_names}"
                    f" by comparing the absolute average performances"
                    f" {features_values} to a defined threshold, which is {self.threshold}.")

        return res_str
