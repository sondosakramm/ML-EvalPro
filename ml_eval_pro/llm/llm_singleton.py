import json

from langchain.output_parsers import StructuredOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.ctransformers import CTransformers
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_experimental.chat_models import Llama2Chat


class LLMSingleton:
    """
    A singleton class for initializing the LLM instance.
    """
    __llm = None

    def __new__(cls, llama_model: str = "llama3"):
        if not cls.__llm:
            cls.__llm = cls.__create_llm(llama_model)
        return cls.__llm

    @classmethod
    def __create_llm(cls, llama_model):
        """
        Initializing the LLM.
        :return: the instantiated model.
        """
        print("Creating the LLM model ...")

        model = ChatOllama(model=llama_model, temperature=0.0)

        return model

    @classmethod
    def execute_prompt(cls, question: str, **kwargs):
        """
        Executing the prompt with the LLM.
        :param question: the input prompt to the LLM with the used variables (if any).
        :return: the response content according to the input prompt.
        """
        print("Executing the prompt ...")

        prompt = ChatPromptTemplate.from_template(question)

        chain = prompt | cls.__llm

        answer = chain.invoke({})
        return json.loads(answer.content)

