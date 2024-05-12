from ml_eval_pro.evaluation_metrics.classification.class_evaluation.accuracy import Accuracy
from ml_eval_pro.evaluation_metrics.classification.class_evaluation.cross_entropy_loss import CrossEntropyLoss
from ml_eval_pro.evaluation_metrics.classification.class_evaluation.f1_score import F1Score
from ml_eval_pro.evaluation_metrics.classification.class_evaluation.fnr import FNR
from ml_eval_pro.evaluation_metrics.classification.class_evaluation.fpr import FPR
from ml_eval_pro.evaluation_metrics.classification.class_evaluation.tnr import TNR
from ml_eval_pro.evaluation_metrics.classification.class_evaluation.tpr import TPR
from ml_eval_pro.evaluation_metrics.classification.probability_evaluation.auc import AUC
from ml_eval_pro.evaluation_metrics.classification.probability_evaluation.expected_calibration_error import ECEMetric
from ml_eval_pro.evaluation_metrics.classification.probability_evaluation.reliability_diagram import \
    ReliabilityDiagram
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.mae import MAE
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.mape import MAPE
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.mean_bias_deviation import MeanBiasDeviation
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.medae import MEDAE
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.r_square_error import RSquare
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.reliability_diagram import Calibration
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.rmse import RMSE


class EvaluatorsFactory:

    @staticmethod
    def get_evaluator(evaluation_metric: str, target, prediction, num_of_classes: int = 2, n_bins: int = 10):
        """
        Create an instance of the appropriate evaluation metric based on the task type.
        :param evaluation_metric: The type of the evaluation metric.
        :param target: The target true/actual values.
        :param prediction: The target predicted values.
        :param num_of_classes: Number of classes (only for classification tasks).
        :param n_bins: the number of bins needed.
        :return: An instance of the evaluation metric.
        """
        try:
            if evaluation_metric == 'MAE':
                return MAE(target, prediction)
            elif evaluation_metric == 'MAPE':
                return MAPE(target, prediction)
            elif evaluation_metric == 'Mean Bias Deviation':
                return MeanBiasDeviation(target, prediction)
            elif evaluation_metric == 'Median Absolute Error':
                return MEDAE(target, prediction)
            elif evaluation_metric == 'R-Squared':
                return RSquare(target, prediction)
            elif evaluation_metric == 'RMSE':
                return RMSE(target, prediction)
            elif evaluation_metric == 'AUC':
                return AUC(target, prediction, num_of_classes, n_bins)
            elif evaluation_metric == 'Expected Calibration Error':
                return ECEMetric(target, prediction, num_of_classes)
            elif evaluation_metric == 'classification reliability evaluation':
                return ReliabilityDiagram(target, prediction, num_of_classes)
            elif evaluation_metric == 'regression reliability evaluation':
                return Calibration(target, prediction, n_bins)
            elif evaluation_metric == 'Accuracy':
                return Accuracy(target, prediction, num_of_classes)
            elif evaluation_metric == 'Cross Entropy Loss':
                return CrossEntropyLoss(target, prediction, num_of_classes)
            elif evaluation_metric == 'F1 Score':
                return F1Score(target, prediction, num_of_classes)
            elif evaluation_metric == 'False Negative Rate':
                return FNR(target, prediction, num_of_classes)
            elif evaluation_metric == 'False Positive Rate':
                return FPR(target, prediction, num_of_classes)
            elif evaluation_metric == 'True Negative Rate':
                return TNR(target, prediction, num_of_classes)
            elif evaluation_metric == 'True Positive Rate':
                return TPR(target, prediction, num_of_classes)
        except NotImplementedError:
            raise NotImplementedError(f'{evaluation_metric} not implemented.')
