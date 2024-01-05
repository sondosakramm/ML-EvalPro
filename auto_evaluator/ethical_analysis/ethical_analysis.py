from auto_evaluator.ethical_analysis.feature_importance.feature_importance_factory import \
    FeatureImportanceFactory
from auto_evaluator.llm.llm_singleton import LLMSingleton


class EthicalAnalysis:
    """
    A class for measuring the ethical analysis for each input feature.
    """
    def __init__(self, model, data, features_description:dict=None, feature_importance_method:str= 'shap'):
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
            LLMSingleton()

            return feature_importance_all_vals, self.__prompt_feature_ethnicity(feature_importance_all_vals[0])

        return feature_importance_all_vals


    def __prompt_feature_ethnicity(self, feature_importance_vals):
        """
        Initializing the model feature importance method needed inputs.
        :param feature_importance_vals: the importance values of each feature.
        :return: the ethical perspective of the top 3 or 5 important features.
        """
        template_feature_importance = """<<SYS>> \nYou are an assistant tasked with answering machine learning related questions.\n <</SYS>>\n\n\
        [INST] You MUST answer the question using ONLY one sentence to illustrate your answer:
        {question} [/INST]"""

        feature_ethics = {}
        feature_size = len(feature_importance_vals.keys())

        if feature_size > 3:
            feature_size = 5

        for i, curr_feature in enumerate(feature_importance_vals):

            if i >= feature_size:
                break

            curr_desc = self.features_description[curr_feature]

            question = f"You have an input feature '{curr_feature}' with description: {curr_desc}. \
            This feature is used in training a machine learning model and it is one of the most important \
            features contributing in predictions. Is this feature '{curr_feature}' ethical and fair to use? and why?"

            feature_ethics[curr_feature] = LLMSingleton.execute_prompt(template_feature_importance, question=question)

        return feature_ethics
