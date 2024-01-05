from auto_evaluator.ethical_analysis.feature_importance.feature_importance_factory import \
    FeatureImportanceFactory
from auto_evaluator.llm.llm_singleton import LLMSingleton


class EthicalAnalysis:
    def __init__(self, model, test_data,features_description:dict=None, feature_importance_method:str='shap'):
        self.model = model
        self.test_data = test_data
        self.features_description = features_description
        self.feature_importance_method = feature_importance_method


    def __call__(self, *args, **kwargs):
        feature_importance_obj = FeatureImportanceFactory.create(self.feature_importance_method,
                                                                 model=self.model,
                                                                 test_data=self.test_data)

        feature_importance_all_vals = feature_importance_obj.calculate()

        if self.features_description:
            LLMSingleton()

            return feature_importance_all_vals, self.__prompt_feature_ethnicity(feature_importance_all_vals[0])

        return feature_importance_all_vals


    def __prompt_feature_ethnicity(self, feature_importance_vals):

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
