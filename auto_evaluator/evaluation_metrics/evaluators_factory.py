from auto_evaluator.evaluation_metrics.classification.class_evaluation.accuracy import Accuracy
from auto_evaluator.evaluation_metrics.classification.class_evaluation.cross_entropy_loss import CrossEntropyLoss
from auto_evaluator.evaluation_metrics.classification.class_evaluation.f1_score import F1Score
from auto_evaluator.evaluation_metrics.classification.class_evaluation.fnr import FNR
from auto_evaluator.evaluation_metrics.classification.class_evaluation.fpr import FPR
from auto_evaluator.evaluation_metrics.classification.class_evaluation.tnr import TNR
from auto_evaluator.evaluation_metrics.classification.class_evaluation.tpr import TPR
from auto_evaluator.evaluation_metrics.classification.probability_evaluation.auc import AUC
from auto_evaluator.evaluation_metrics.classification.probability_evaluation.expected_calibration_error import ECEMetric
from auto_evaluator.evaluation_metrics.classification.probability_evaluation.reliability_diagram import \
    ReliabilityDiagram
from auto_evaluator.evaluation_metrics.regression.evaluation_metrics.mae import MAE
from auto_evaluator.evaluation_metrics.regression.evaluation_metrics.mape import MAPE
from auto_evaluator.evaluation_metrics.regression.evaluation_metrics.mean_bias_deviation import MeanBiasDeviation
from auto_evaluator.evaluation_metrics.regression.evaluation_metrics.medae import MEDAE
from auto_evaluator.evaluation_metrics.regression.evaluation_metrics.r_square_error import RSquare
from auto_evaluator.evaluation_metrics.regression.evaluation_metrics.reliability_diagram import Calibration
from auto_evaluator.evaluation_metrics.regression.evaluation_metrics.rmse import RMSE


class EvaluatorsFactory:

    @staticmethod
    def get_evaluator(evaluation_metric: str, target, prediction, num_of_classes: int = 2, n_bins: int = 5,
                      display: bool = False):
        """
        Create an instance of the appropriate evaluation metric based on the task type.
        :param evaluation_metric: The type of the evaluation metric.
        :param target: The target true/actual values.
        :param prediction: The target predicted values.
        :param num_of_classes: Number of classes (only for classification tasks).
        :param n_bins: the number of bins needed.
        :param display: display the reliability diagram plot.
        :return: An instance of the evaluation metric.
        """
        try:
            if evaluation_metric == 'mae':
                return MAE(target, prediction)
            elif evaluation_metric == 'mape':
                return MAPE(target, prediction)
            elif evaluation_metric == 'mean bias deviation':
                return MeanBiasDeviation(target, prediction)
            elif evaluation_metric == 'median absolute error':
                return MEDAE(target, prediction)
            elif evaluation_metric == 'r square':
                return RSquare(target, prediction)
            elif evaluation_metric == 'rmse':
                return RMSE(target, prediction)
            elif evaluation_metric == 'auc':
                return AUC(target, prediction, num_of_classes, n_bins)
            elif evaluation_metric == 'expected calibration error':
                return ECEMetric(target, prediction, num_of_classes, n_bins)
            elif evaluation_metric == 'classification reliability evaluation':
                return ReliabilityDiagram(target, prediction, num_of_classes, n_bins, display)
            elif evaluation_metric == 'regression reliability evaluation':
                return Calibration(target, prediction, n_bins)
            elif evaluation_metric == 'accuracy':
                return Accuracy(target, prediction, num_of_classes)
            elif evaluation_metric == 'cross entropy loss':
                return CrossEntropyLoss(target, prediction, num_of_classes)
            elif evaluation_metric == 'f1 score':
                return F1Score(target, prediction, num_of_classes)
            elif evaluation_metric == 'false negative rate':
                return FNR(target, prediction, num_of_classes)
            elif evaluation_metric == 'false positive rate':
                return FPR(target, prediction, num_of_classes)
            elif evaluation_metric == 'true negative rate':
                return TNR(target, prediction, num_of_classes)
            elif evaluation_metric == 'true positive rate':
                return TPR(target, prediction, num_of_classes)
        except NotImplementedError:
            raise NotImplementedError(f'{evaluation_metric} not implemented.')
