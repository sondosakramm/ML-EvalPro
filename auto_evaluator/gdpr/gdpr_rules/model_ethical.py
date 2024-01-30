import string

from auto_evaluator.gdpr.gdpr_compliance import GdprCompliance
from auto_evaluator.llm.llm_singleton import LLMSingleton


class ModelEthical(GdprCompliance):

    def __get_unethical_features(self):

        """
        Initializing the model feature description method needed inputs.
        :return: a list of the unethical features.
        """

        LLMSingleton()
        template_feature_importance = """<<SYS>> \nYou are an assistant tasked with answering machine learning related questions.\n <</SYS>>\n\n\
        [INST] Answer each question independently. You MUST answer the question using ONLY one sentence to illustrate your answer:
        {question} [/INST]"""

        feature_unethical = {'feature': [],
                             'reason': []}

        for curr_feature in self.features_description.keys():

            curr_desc = self.features_description[curr_feature]

            question = f"You have an input feature '{curr_feature}' with description: {curr_desc}. \
            This feature is used in training a machine learning model and it is one of the most important \
            features contributing in predictions. Is this feature '{curr_feature}' ethical and fair to use? and why?\
            If the input feature is ethical, say the word 'True'. If the input feature is not ethical, say the word 'False'."


            single_feature_ethics = LLMSingleton.execute_prompt(template_feature_importance, question=question)

            if not eval(single_feature_ethics.split()[0][:-1]):
                feature_unethical['feature'].append(curr_feature)
                exclude_set = {"true", "false"}
                translator = str.maketrans("", "", string.punctuation)
                result_text = ' '.join(
                    word for word in single_feature_ethics.split() if
                    word.lower().translate(translator) not in exclude_set)
                feature_unethical['reason'].append(result_text)

        return feature_unethical

    def __str__(self):
        summary_str = f'{5 * "*"} Model Ethical {5 * "*"}\n'
        feature_unethical = self.__get_unethical_features()
        for i in range(len(feature_unethical['feature'])):
            summary_str += (f'Feature {feature_unethical["feature"][i]} is not ethical because '
                            f'{feature_unethical["reason"][i]}\n')
        return summary_str
