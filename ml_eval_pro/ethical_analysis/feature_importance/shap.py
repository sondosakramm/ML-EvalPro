import numpy as np
import pandas as pd
import shap

from ml_eval_pro.ethical_analysis.feature_importance.feature_importance import FeatureImportance


class SHAP(FeatureImportance):
    """
    A class for calculating feature importance with Shapley values.
    """

    def calculate(self):
        """
        Calculating the feature importance with SHAP method.
        :return: The feature importance values sorted in descending order, SHAP values, and the expected values for
                 each feature.
        """
        explainer = shap.Explainer(self.model.predict, shap.utils.sample(self.data, int(self.data.shape[0] * 0.1)))
        shap_values = explainer(self.data).values
        feature_importance = self.__calculate__shap_abs_mean(shap_values)
        return feature_importance

    def __calculate__shap_abs_mean(self, shap_values):
        """
        Calculating each feature importance with the absolute mean of the shapley values of each feature.
        :param shap_values: the shapley values for each feature.
        :return: The feature importance values sorted in descending order.
        """
        abs_shap = np.abs(shap_values)
        mean_abs_shap = np.mean(abs_shap, axis=0)
        summary_df = pd.DataFrame(mean_abs_shap, index=self.data.columns, columns=['Mean_ABS_SHAP'])
        summary_df.sort_values(by='Mean_ABS_SHAP', ascending=False, inplace=True)
        return list(summary_df.to_dict().values())[0]
