from sklearn.metrics import accuracy_score

from ml_eval_pro.evaluation_metrics.classification.class_evaluation.class_evaluation import ClassClassification


class Accuracy(ClassClassification):
    """
    A class for evaluating the model using Accuracy evaluation metric.
    """

    def measure(self):
        """
        Measure the accuracy of the model.
        :return: the accuracy score of the model.
        """
        return accuracy_score(self.target, self.prediction) * 100
