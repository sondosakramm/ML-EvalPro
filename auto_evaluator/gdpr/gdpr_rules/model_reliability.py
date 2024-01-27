from auto_evaluator.evaluation_metrics.evaluators_factory import EvaluatorsFactory
from auto_evaluator.gdpr.gdpr_compliance import GdprCompliance


class ModelReliability(GdprCompliance):
    def __str__(self):
        self.prediction = self.model.predict(self.X_test)
        summary_str = f'{5*"*"} Model Reliability {5*"*"}\n'
        if self.problem_type == 'classification':
            evaluator = EvaluatorsFactory.get_evaluator("expected calibration error",
                                                        self.y_test, self.prediction,
                                                        self.num_of_classes, self.n_bins)
            ece = evaluator.measure()
            summary_str += (f'The model reliability evaluation showed a mismatch between the model confidence level '
                            f'and the accuracy of its predictions by {ece * 100: .5f}% on average.\n')

            return summary_str
        elif self.problem_type == 'regression':
            summary_str += f'Reliability Regression Graph'
            # --------------------- Graph ---------------------
            return summary_str
        summary_str += f'Problem type is not supported.'
        return summary_str
