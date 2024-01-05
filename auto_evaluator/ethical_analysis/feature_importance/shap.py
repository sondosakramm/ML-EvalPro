import numpy as np
import pandas as pd
import shap

from auto_evaluator.ethical_analysis.feature_importance.feature_importance import FeatureImportance


class SHAP(FeatureImportance):
    def calculate(self):
        explainer = shap.Explainer(self.model)
        shap_vals = explainer.shap_values(self.test_data)
        feature_importance = self.__calculate__shap_abs_mean(shap_vals)
        return feature_importance, shap_vals, explainer.expected_value

    def __calculate__shap_abs_mean(self, shap_values):
        mean_abs_shap = np.mean(np.abs(shap_values), axis=1)
        summary_df = pd.DataFrame(mean_abs_shap.sum(axis=0), index=self.test_data.columns, columns=['Mean_ABS_SHAP'])
        summary_df.sort_values(by='Mean_ABS_SHAP', ascending=False, inplace=True)
        return list(summary_df.to_dict().values())[0]