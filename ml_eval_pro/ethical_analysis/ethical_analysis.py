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
                    EthicalAnalysis.prompt_feature_ethnicity(self.features_description, feature_importance_all_vals))

        return (feature_importance_all_vals,
                "Unable to address ethical concerns at this time, as no description or details have been provided.")

    @classmethod
    def prompt_feature_ethnicity(cls, features_description, feature_importance_vals):
        """
        Prompting ethnicity of the input features.
        :param features_description: the description of each feature.
        :param feature_importance_vals: the importance values of each feature.
        :return: the ethical perspective of the top 3 or 5 important features.
        :return: a list of the unethical features.
        """
        LLMSingleton()
        template_feature_importance = """<<SYS>> \nYou are an assistant tasked with answering machine learning related questions.\n <</SYS>>\n\n\
        [INST] Answer each question independently. You MUST answer the question using ONLY one sentence to illustrate your answer:
        {question} [/INST]"""

        feature_unethical = []

        feature_size = len(feature_importance_vals.keys())

        if feature_size > 5:
            feature_size = 5

        for i, curr_feature in enumerate(feature_importance_vals):
            print(f"Evaluating the feature {curr_feature} ...")

            if i >= feature_size:
                break

            curr_desc = features_description[curr_feature]

            question = f"You have an input feature '{curr_feature}' with description: {curr_desc}. \
            This feature is used in training a machine learning model and it is one of the most important \
            features contributing in predictions. Is this feature '{curr_feature}' ethical and fair to use? and why?\
            If the input feature is ethical, say the word 'True'. If the input feature is not ethical, say the word 'False'."

            single_feature_ethics = LLMSingleton.execute_prompt(template_feature_importance, question=question)

            if not eval(single_feature_ethics.split()[0][:-1]):
                feature_unethical.append(f'{curr_feature}: {single_feature_ethics.split("False. ")[1]}')

            if len(feature_unethical) == 0:
                return f"No features of the most {feature_size} important features was unethical."

        return (f"Out of the most {feature_size}, the following features were unethical:\n" +
                '\n'.join(feature_unethical))
