import pandas as pd

from ml_eval_pro.ethical_analysis.feature_importance.feature_importance_factory import \
    FeatureImportanceFactory
from ml_eval_pro.llm.llm_singleton import LLMSingleton


class EthicalAnalysis:
    """
    A class for measuring the ethical analysis for each input feature.
    """

    def __init__(self, model, data, features_description: dict, dataset_context: str,
                 feature_importance_method: str = 'shap'):
        """
        Initializing the ethical analysis needed inputs.
        :param model: the model.
        :param data: the dataset containing all the features.
        :param features_description: a short description for each feature.
        :param dataset_context: a description of the dataset context.
        :param feature_importance_method: the method used to measure the feature importance.
        """
        self.model = model
        self.data = data
        self.features_description = features_description
        self.dataset_context = dataset_context
        self.feature_importance_method = feature_importance_method

    def __call__(self, *args, **kwargs):
        """
        Executing the ethical analysis on features.
        :return: the importance value of each feature and the ethical perspective of the most important features.
        """
        feature_importance_obj = FeatureImportanceFactory.create(self.feature_importance_method,
                                                                 model=self.model,
                                                                 data=self.data)

        feature_importance_all_vals = feature_importance_obj.calculate()

        return (feature_importance_all_vals,
                EthicalAnalysis.prompt_feature_ethnicity(self.features_description,
                                                         self.dataset_context,
                                                         self.data))

    @classmethod
    def prompt_feature_ethnicity(cls, features_description, dataset_context: str, dataset: pd.DataFrame):
        """
        Prompting ethnicity of the input features.
        :param features_description: the description of each feature.
        :param dataset_context: a description of the dataset context.
        :param dataset: the dataset given.
        :return: the ethical perspective of the given features.
        """
        dataset_sample = dataset.iloc[:10, :] if dataset.shape[0] >= 10 else dataset

        dataset_sample_str = dataset_sample.to_dict('list').__str__()

        unethical_features = cls.get_unethical_features(features_description, dataset_context,
                                                        dataset_sample_str[1:len(dataset_sample_str)-1])

        if len(unethical_features) == 0:
            return f"No unethical feature was detected."

        res_str = ""
        for unethical_feature in unethical_features.keys():
            res_str += f'\n{unethical_feature}: {unethical_features[unethical_feature]}'
        return (f"Out of the features descriptions provided, the features below were unethical for the following "
                f"reasons:" + res_str)

    @classmethod
    def get_unethical_features(cls, features_descriptions: dict, dataset_context: str, dataset_sample_str: str) -> dict:
        """
         Prompting the unethical features given their description with LLMs.
         :param features_descriptions: the description of each feature.
         :param dataset_context: a description of the dataset context.
         :param dataset_sample_str: a sample from the dataset.
         :return: the unethical features and the reason of being unethical.
         """
        LLMSingleton()

        features_str = features_descriptions.__str__()

        question = f"Given the dataset context: {dataset_context} and the following sample of each feature: {dataset_sample_str}, \
        along with the descriptions {features_str[1:len(features_str) - 1]}. They are used in training a machine learning model \
        and they are one of the most important features contributing in predictions. In this context, determine which feature(s) are unethical to use and a short reason for the answer. \
        Answer the question with json format only where the key is the feature and the value is the reason. The answer should not include any additional notes."

        unethical_features = LLMSingleton.execute_prompt(question)

        return unethical_features
