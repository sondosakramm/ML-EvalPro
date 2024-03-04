from ml_eval_pro.summary.summary_generator import SummaryGenerator
from ml_eval_pro.variance.model_variance_by_test_data import ModelVarianceByTestData
from ml_eval_pro.variance.model_variance_by_train_test_data import ModelVarianceByTrainTestData


class VarianceSummary(SummaryGenerator):
    def __init__(self, variance):
        self.variance = variance

    def get_summary(self):
        summary = ''
        if isinstance(self.variance, ModelVarianceByTestData):
            features_exceed_threshold = self.variance.get_diff()  # This now contains feature names
            if len(features_exceed_threshold) > 0:
                summary += f'High predictions variance is detected when doing small perturbations on features '
                i = 0
                for column in self.variance.X_test_with_features_name.columns:
                    if i in features_exceed_threshold:
                        summary += f'{column}, '
                    i += 1
                summary += (f'\nwhich suggests model overfitting on the training set on these specific features '
                            f'or the high importance of these features.')
            else:
                summary += 'High predictions variance is NOT detected when doing small perturbations on features '
        elif isinstance(self.variance, ModelVarianceByTrainTestData):
            if self.variance.calculate_variance() > self.variance.yaml_reader.get('variance')['threshold']:
                summary += f'High variance is detected.'
            else:
                summary += f'No high variance is detected.'
        return summary
