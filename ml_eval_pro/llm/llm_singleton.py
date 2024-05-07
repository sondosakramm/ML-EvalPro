from langchain.output_parsers import StructuredOutputParser
from langchain_community.llms.ctransformers import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain_experimental.chat_models import Llama2Chat


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
        llm = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGUF',
                            model_file='llama-2-7b-chat.Q4_0.gguf',
                            temperature=0.0, max_new_tokens=128, batch_size=256, top_k=1)

        model = Llama2Chat(llm=llm)

        return model

    @classmethod
    def execute_prompt(cls, question: str, response_schema: list, **kwargs):
        """
        Executing the prompt with the LLM.
        :param question: the input prompt to the LLM with the used variables (if any).
        :param response_schema: the schema defined by the user that the prompt should follow.
        :return: the response content according to the input prompt.
        """
        print("Executing the prompt ...")

        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser.get_format_instructions()

        prompt = PromptTemplate(
            template="You are an assistant tasked with answering machine learning related questions. \
            You must answer with ONLY one sentence. Answer the user's question without any assumptions.\n{format_instructions}\n{question}",
            input_variables=['question'],
            partial_variables={"format_instructions": format_instructions}
        )

        llm_prompt_input = prompt.format_prompt(question=question)
        prompt_output = cls.__llm(llm_prompt_input.to_messages())

        return output_parser.parse(prompt_output.content)
