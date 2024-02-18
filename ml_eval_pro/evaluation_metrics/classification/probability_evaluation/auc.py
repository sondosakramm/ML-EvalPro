from sklearn.metrics import roc_auc_score

from ml_eval_pro.evaluation_metrics.classification.probability_evaluation.probability_evaluation import \
    ProbabilityClassification


class AUC(ProbabilityClassification):
    """
    A class for evaluating the model using the Area Under the ROC Curve evaluation metric.
    """
    def measure(self):
        """
        Measure the area under the ROC curve of the model.
        :return: the ROC AUC score of the model.
        """
        if self.num_of_classes == 2:
            return roc_auc_score(self.target, self.prediction_prob[:,1])

        return roc_auc_score(self.target, self.prediction_prob, multi_class="ovo")