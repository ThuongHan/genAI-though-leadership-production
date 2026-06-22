from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import os
from dotenv import load_dotenv
from Generator.utils.llm.base import BaseLLM

# gpt models: ["gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o"]


class GPTLLM(BaseLLM):

    def __init__(self, model: str = "gpt-5.1"):
        load_dotenv("Generator/secrets/uva_api_key.env", override=True)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")

        self.model = model

        self.llm = ChatOpenAI(
            model=self.model,
            api_key=SecretStr(api_key),
        )

    def invoke(self, prompt: str):
        return self.llm.invoke(prompt)
    
if __name__ == "__main__":
    query = "Are you present? (Yes/No)"

    gpt4_1 = GPTLLM("gpt-4.1")
    result = gpt4_1.invoke(query)
    print("GPT4.1: ", result.content)

    gpt5_1 = GPTLLM()
    result = gpt5_1.invoke(query)
    print("GPT5.1: ", result.content)
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    for model in client.models.list().data:
        print(model.id)

# python3 -m Generator.utils.llm.gpt