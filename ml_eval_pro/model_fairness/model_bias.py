import os

import pandas as pd

from ml_eval_pro.configuration_manager.configuration_reader.yaml_reader import YamlReader
from ml_eval_pro.model_fairness.model_fairness import ModelFairness


class ModelBias(ModelFairness):
    """
    A class for measuring the model bias.
    """

    def get_features_names(self):
        return self.data.columns.tolist()

    def execute_post_steps(self, features_abs_avg_performance: pd.DataFrame):
        avg_eval_metrics = features_abs_avg_performance.mean(axis=0)

        threshold = YamlReader(os.path.join(os.path.curdir,
                                            "ml_eval_pro",
                                            "config_files",
                                            "system_config.yaml")).get('thresholds')['bias_threshold']

        return avg_eval_metrics[avg_eval_metrics >= threshold].to_dict()
