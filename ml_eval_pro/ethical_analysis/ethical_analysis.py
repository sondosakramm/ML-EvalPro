from langchain.output_parsers import ResponseSchema

from ml_eval_pro.ethical_analysis.feature_importance.feature_importance_factory import \
    FeatureImportanceFactory
from ml_eval_pro.llm.llm_singleton import LLMSingleton


class EthicalAnalysis:
    """
    A class for measuring the ethical analysis for each input feature.
    """

    def __init__(self, model, data, features_description: dict = None, feature_importance_method: str = 'shap'):
        """
        Initializing the ethical analysis needed inputs.
        :param model: the model.
        :param data: the dataset containing all the features.
        :param features_description: a short description for each feature.
        :param feature_importance_method: the method used to measure the feature importance.
        """
        self.model = model
        self.data = data
        self.features_description = features_description
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

        if self.features_description:
            return (feature_importance_all_vals,
                    EthicalAnalysis.prompt_feature_ethnicity(self.features_description))

        return (feature_importance_all_vals,
                "Unable to address ethical concerns at this time, as no description or details have been provided.")

    @classmethod
    def prompt_feature_ethnicity(cls, features_description):
        """
        Prompting ethnicity of the input features.
        :param features_description: the description of each feature.
        :return: the ethical perspective of the given features.
        """
        unethical_features = cls.get_unethical_features(features_description)

        if len(unethical_features) == 0:
            return f"No unethical feature was detected."

        res_str = ""
        for unethical_feature in unethical_features.keys():
            res_str += f'\n{unethical_feature}: {unethical_features[unethical_feature]}'
        return (f"Out of the features descriptions provided, the features below were unethical for the following "
                f"reasons:" + res_str)

    @classmethod
    def get_unethical_features(cls, features_descriptions: dict) -> dict:
        """
         Prompting the unethical features given their description with LLMs.
         :param features_descriptions: the description of each feature.
         :return: the unethical features and the reason of being unethical.
         """
        LLMSingleton()
        unethical_features = {}
        for feature in features_descriptions.keys():
            print(f"Evaluating the feature {feature} ...")
            desc = features_descriptions[feature]

            question = f"The feature '{feature}' with description '{desc}' is used in training a machine learning model and it is one of the most important \
                    features contributing in predictions. Is it ethical and fair to use?"

            response_schema = [
                ResponseSchema(name="answer", description="answer to the user's question", type='boolean'),
                ResponseSchema(name="reason", description="short reason of the answer of the user's question")
            ]

            curr_res = LLMSingleton.execute_prompt(question, response_schema)

            if not curr_res['answer']:
                unethical_features[feature] = curr_res['reason']

        return unethical_features
