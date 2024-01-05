from os.path import expanduser

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_experimental.chat_models import Llama2Chat
from langchain.prompts import ChatPromptTemplate

class LLMSingleton:
    """
    A singleton class for initializing the LLM instance.
    """
    __llm = None

    def __new__(cls):
        if not cls.__llm:
            cls.__llm = cls.__create_llm("llama-2-7b-chat.Q4_0.gguf")
        return cls.__llm

    @classmethod
    def __create_llm(cls, model_path_str: str):
        """
        Initializing the LLM.
        :param model_path_str: the path to the LLM model file (.gguf extension) for LLamaCpp.
        :return: the instantiated model.
        """
        model_path = expanduser(model_path_str)

        llm = LlamaCpp(
            model_path=model_path,
            n_batch=32,
            n_ctx=1024,
            streaming=False
        )

        model = Llama2Chat(llm=llm)

        return model

    @classmethod
    def execute_prompt(cls, prompt: str, **kwargs):
        """
        Executing the prompt with the LLM.
        :param prompt: the input prompt to the LLM with the used variables (if any).
        :return: the response content according to the input prompt.
        """
        prompt_template = ChatPromptTemplate.from_template(prompt)
        msg_format = prompt_template.format_messages(**kwargs)

        return cls.__llm(msg_format).content
