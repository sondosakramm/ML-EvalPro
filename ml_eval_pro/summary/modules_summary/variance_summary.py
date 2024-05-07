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
            summary += (f"To find out how sensitive the model is to these changes, small changes in the input data "
                        f"were simulated. We change the values of categorical features to other values within their "
                        f"unique values and slightly adjust the values of numerical features. Next, evaluate the "
                        f"model's predictions considering all of these perturbed inputs.\n")
            if len(features_exceed_threshold) > 0:
                summary += f'High predictions variance is detected when doing small perturbations on features \n'
                i = 0
                for column in self.variance.X_test_with_features_name.columns:
                    if i in features_exceed_threshold:
                        summary += f'{column}, '
                    i += 1
                summary += (f'which suggests model overfitting on the training set on these specific features '
                            f'or the high importance of these features.')
            else:
                summary += 'High predictions variance is NOT detected when doing small perturbations on features '
        elif isinstance(self.variance, ModelVarianceByTrainTestData):
            summary += (f"First evaluation for the model's performance on both the data it was trained on (X_train) and"
                        f" the unseen data (X_test). By comparing the model's predictions on these two sets, "
                        f"the difference in performance is calculate.\n")
            if self.variance.calculate_variance() > self.variance.yaml_reader.get('variance')['threshold']:
                summary += (f"Since this difference exceeds the threshold = "
                            f"{self.variance.yaml_reader.get('variance')['threshold']}, "
                            f"it indicates that the model is not generalizing well and has high variance.")
            else:
                summary += (f"Since this difference does not exceeds the threshold = "
                            f"{self.variance.yaml_reader.get('variance')['threshold']}, "
                            f"it indicates that the model is generalizing well and no high variance is detected.")
        return summary
