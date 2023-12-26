import pandas as pd

from auto_evaluator.bias.feature_bias.feature_bias import FeatureBias
from sklearn.cluster import KMeans

class NumericalBias(FeatureBias):
    """
    A class for the numerical bias of an input feature.
    """
    def __init__(self, model, target: pd.Series, feature: pd.Series,
                 performance_metric: str = 'accuracy', significance: float = 0.05,
                 no_of_clusters: int = 5):
        """
        Initializing the feature bias needed inputs.
        :param model: the model.
        :param target: the target prediction values.
        :param feature: the input feature values.
        :param performance_metric: the performance metric used for measuring the bias.
        :param significance: the significance value to measure bias.
        :param no_of_clusters: the number of binning ranges of the numerical feature.
        """
        super().__init__(model, target, feature, performance_metric, significance)
        self.no_of_clusters = no_of_clusters


    def __call__(self, *args, **kwargs):
        """
        Calculating the numerical bias of a single feature.
        :return: the average absolute performances and a boolean indicating if the model is biased according to that feature.
        """

        # Binning the feature numerical ranges by KMeans clustering by the input number of clusters.
        kmeans_clustering = KMeans(n_clusters=self.no_of_clusters)
        clusters = kmeans_clustering.fit_predict(self.feature)

        eval_metrics = self._calculate_categorical_metrics(clusters)
        pairwise_diff, avg_abs_performance = FeatureBias._calculate_average_absolute_performance(eval_metrics)

        return avg_abs_performance, avg_abs_performance >= self.significance


