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
from ml_eval_pro.evaluation_metrics.evaluation_metric import EvaluationMetric
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.mae import MAE
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.mape import MAPE
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.mean_bias_deviation import MeanBiasDeviation
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.medae import MEDAE
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.r_square_error import RSquare
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.reliability_diagram import Calibration
from ml_eval_pro.evaluation_metrics.regression.evaluation_metrics.rmse import RMSE


class EvaluatorsFactory:
    _factory_supported_classes = {
        'MAE': MAE,
        'MAPE': MAPE,
        'Mean Bias Deviation': MeanBiasDeviation,
        'Median Absolute Error': MEDAE,
        'R-Squared': RSquare,
        'RMSE': RMSE,
        'AUC': AUC,
        'Expected Calibration Error': ECEMetric,
        'classification reliability evaluation': ReliabilityDiagram,
        'regression reliability evaluation': Calibration,
        'Accuracy': Accuracy,
        'Cross Entropy Loss': CrossEntropyLoss,
        'F1 Score': F1Score,
        'False Negative Rate': FNR,
        'False Positive Rate': FPR,
        'True Negative Rate': TNR,
        'True Positive Rate': TPR
    }

    @classmethod
    def create(cls, metric: str, *args, **kwargs) -> EvaluationMetric:
        if metric in cls._factory_supported_classes:
            evaluator_class = cls._factory_supported_classes.get(metric)
            return evaluator_class(*args, **kwargs)
        else:
            raise NotImplementedError(f'{metric} is not implemented.')
