import pandas as pd

from ml_eval_pro.model_fairness.model_fairness import ModelFairness


class ModelBias(ModelFairness):
    """
    A class for measuring the model bias.
    """

    def __init__(self, model, model_type: str, data: pd.DataFrame, target: pd.Series, evaluation_metrics: [str],
                 threshold: float = 0.15):
        """
        Initializing the model fairness needed inputs.
        :param model: the model.
        :param model_type: the model type.
        :param data: the dataset containing all the features.
        :param target: the target values.
        :param evaluation_metrics: the evaluation metric used to measure fairness.
        :param threshold: The bias threshold for assessing model bias.
        """
        self.threshold = threshold
        super().__init__(model, model_type, data, target, evaluation_metrics)


    def get_features_names(self):
        return self.data.columns.tolist()

    def execute_post_steps(self, features_abs_avg_performance: pd.DataFrame):
        avg_eval_metrics = features_abs_avg_performance.mean(axis=0)

        return avg_eval_metrics[avg_eval_metrics >= self.threshold].to_dict()
