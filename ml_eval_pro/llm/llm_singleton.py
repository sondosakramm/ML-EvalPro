from langchain_community.llms.ctransformers import CTransformers
from langchain_experimental.chat_models import Llama2Chat
from langchain.prompts import ChatPromptTemplate

class LLMSingleton:
    """
    A singleton class for initializing the LLM instance.
    """
    __llm = None

    def __new__(cls):
        if not cls.__llm:
            cls.__llm = cls.__create_llm()
        return cls.__llm

    @classmethod
    def __create_llm(cls):
        """
        Initializing the LLM.
        :return: the instantiated model.
        """
        print("Creating the LLM model ...")
        llm = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGUF', model_file='llama-2-7b-chat.Q4_0.gguf')

        model = Llama2Chat(llm=llm)

        return model

    @classmethod
    def execute_prompt(cls, prompt: str, **kwargs):
        """
        Executing the prompt with the LLM.
        :param prompt: the input prompt to the LLM with the used variables (if any).
        :return: the response content according to the input prompt.
        """
        print("Executing the prompt ...")
        prompt_template = ChatPromptTemplate.from_template(prompt)
        msg_format = prompt_template.format_messages(**kwargs)

        return cls.__llm(msg_format).content
